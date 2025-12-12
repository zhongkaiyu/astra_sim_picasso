/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
ethane/
***/

#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/MeshAllGather.hh"

#include <cmath>
#include <iostream>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/MemBus.hh"

using namespace AstraSim;

MeshAllGather::MeshAllGather(ComType type,
                             int id,
                             Mesh2DTopology* mesh_topology,
                             uint64_t data_size)
    : Algorithm() {
    this->comType = type;
    this->id = id;
    this->logical_topo = mesh_topology;
    this->mesh_topo = mesh_topology;
    this->data_size = data_size;
    this->total_nodes = mesh_topology->get_total_nodes();
    
    // Calculate maximum hops needed (Manhattan distance across diagonal)
    int width = mesh_topology->get_width();
    int height = mesh_topology->get_height();
    this->max_hops = (width - 1) + (height - 1);
    
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    logger->info("Initializing MeshAllGather: id={}, total_nodes={}, width={}, height={}, max_hops={}", 
                 id, total_nodes, width, height, max_hops);
    
    this->parallel_reduce = 1;
    this->total_packets_sent = 0;
    this->total_packets_received = 0;
    this->free_packets = 0;
    this->zero_latency_packets = 0;
    this->non_zero_latency_packets = 0;
    this->toggle = false;
    this->name = Name::Mesh2D;
    this->transmition = MemBus::Transmition::Usual;
    
    // For AllGather on N nodes in a mesh, we need to send more times than N-1 due to flooding
    // Max hops across diagonal * number of phases needed for complete gather
    // For flooding: need max_hops rounds * num_neighbors packets per round = 6 * 4 max
    // Set stream_count high enough to allow all flooding rounds + extra buffer
    this->stream_count = max_hops * 4;  // Conservative: allow many rounds
    this->max_count = 1;  // Start with 1 to allow initial packet release
    this->remained_packets_per_message = 1;
    this->remained_packets_per_max_count = 1;
    
    // AllGather: final data size is original data size times number of nodes
    this->final_data_size = data_size * total_nodes;
    this->msg_size = data_size;
    
    // Mark our own ID as received (we have our own data)
    received_from_npus.insert(id);

  // Build neighbor list
  neighbors.clear();
  for (int d = 0; d < 4; d++) {
    int n = mesh_topo->get_neighbor(id, (Mesh2DTopology::Direction)d);
    if (n != -1) neighbors.push_back(n);
  }
  num_neighbors = (int)neighbors.size();
  current_round = 0;
  pending_receives_this_round = 0;
}

int MeshAllGather::get_non_zero_latency_packets() {
    // For AllGather on N nodes, each NPU needs to exchange with neighbors multiple times
    // to propagate all data across the mesh
    return (total_nodes - 1) * parallel_reduce * 1;
}

void MeshAllGather::forward_to_neighbors(int received_from_npu) {
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    
    // Get all neighbors
    int directions[] = {
        (int)Mesh2DTopology::Direction::East,
        (int)Mesh2DTopology::Direction::West,
        (int)Mesh2DTopology::Direction::North,
        (int)Mesh2DTopology::Direction::South
    };
    
    int forwarded_count = 0;
    for (int dir : directions) {
        int neighbor = mesh_topo->get_neighbor(id, (Mesh2DTopology::Direction)dir);
        
        if (neighbor != -1 && neighbor != received_from_npu) {
            // Forward to this neighbor
            logger->debug("NPU {} forwarding to neighbor NPU {}", id, neighbor);
            
            // Create a packet to forward to this neighbor
            MyPacket packet(stream->current_queue_id, id, neighbor);
            packets.push_back(packet);
            packets.back().sender = nullptr;
            locked_packets.push_back(&packets.back());
            
            processed = false;
            send_back = false;
            NPU_to_MA = false;
            
            forwarded_count++;
        }
    }
    
    if (forwarded_count > 0) {
        logger->info("NPU {} forwarded packet to {} neighbors", id, forwarded_count);
        // Release the forwarded packets so they actually get sent
        release_packets();
    }
}

// Schedule sends/recvs to all neighbors for the current round.
// We use PacketBundle path to handle resource accounting, and then
// issue a matching send/recv like Ring for each neighbor.
void MeshAllGather::schedule_round() {
  if ((int)received_from_npus.size() == total_nodes) {
    // Done
    return;
  }
  // Emit one packet per neighbor, each released separately to get one General per send
  pending_receives_this_round += num_neighbors;
  auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
  logger->info("NPU {} schedule_round round={}: scheduling {} neighbors", 
               id, current_round, num_neighbors);
  for (int neighbor : neighbors) {
    packets.push_back(MyPacket(stream->current_queue_id, neighbor, neighbor));
    packets.back().sender = nullptr;
    locked_packets.push_back(&packets.back());
    processed = false;
    send_back = false;
    NPU_to_MA = (current_round == 0);
    process_max_count();
  }
}

void MeshAllGather::release_packets() {
    for (auto packet : locked_packets) {
        packet->set_notifier(this);
    }
    if (NPU_to_MA == true) {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_MA();
    } else {
        (new PacketBundle(stream->owner, stream, locked_packets, processed,
                          send_back, msg_size, transmition))
            ->send_to_NPU();
    }
    locked_packets.clear();
}

void MeshAllGather::process_stream_count() {
    if (remained_packets_per_message > 0) {
        remained_packets_per_message--;
    }
    if (id == 0) {
        auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
        logger->info("NPU 0 progress: stream_count={}, received_from={}/{}", 
                     stream_count, received_from_npus.size(), total_nodes);
    }
}

void MeshAllGather::process_max_count() {
    if (remained_packets_per_max_count > 0) {
        remained_packets_per_max_count--;
    }
    if (remained_packets_per_max_count == 0) {
        max_count--;
        release_packets();
        remained_packets_per_max_count = 1;
    }
}

void MeshAllGather::reduce() {
    process_stream_count();
    packets.pop_front();
    free_packets--;
    total_packets_sent++;
}

bool MeshAllGather::iteratable() {
  auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
  logger->info("NPU {} iteratable: round={}/{}, pending={}, packets={}, free={}", 
               id, current_round, max_hops, pending_receives_this_round, packets.size(), free_packets);
  // Terminate flooding after max_hops rounds when all recvs arrived and queues drained
  // For mesh flooding, we continue until we've done all rounds and all local
  // packets have been processed. The streaming framework will clean up when 
  // no more events are generated.
  bool all_rounds_done = (current_round >= max_hops);
  bool no_pending = (pending_receives_this_round == 0);
  bool queues_empty = packets.empty() && locked_packets.empty();
  
  if (all_rounds_done && no_pending && queues_empty) {
    logger->info("NPU {} reached completion: rounds={}, neighbors={}", id, current_round, num_neighbors);
    // Don't call exit() explicitly - let framework clean up to avoid resource leaks
    return false;
  }
    return true;
}

void MeshAllGather::insert_packet(Callable* sender) {
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    logger->info("NPU {} insert_packet called", id);
    
    if (zero_latency_packets == 0 && non_zero_latency_packets == 0) {
        zero_latency_packets = parallel_reduce * 1;  // Like Ring
        non_zero_latency_packets = get_non_zero_latency_packets();
        toggle = !toggle;
        logger->info("NPU {} initialized packets: zero={}, non_zero={}", 
                     id, zero_latency_packets, non_zero_latency_packets);
    }
    
    // Pick first available neighbor (simplified mesh exchange)
    int neighbor = -1;
    for (int d = 0; d < 4; d++) {
        int n = mesh_topo->get_neighbor(id, (Mesh2DTopology::Direction)d);
        if (n != -1) {
            neighbor = n;
            break;
        }
    }
    
    if (neighbor == -1) {
        return;  // No neighbors (shouldn't happen)
    }
    
    if (zero_latency_packets > 0) {
        // Create packet exactly like Ring does
        packets.push_back(MyPacket(stream->current_queue_id, neighbor, neighbor));
        packets.back().sender = sender;
        locked_packets.push_back(&packets.back());
        processed = false;
        send_back = false;
        NPU_to_MA = true;
        process_max_count();
        zero_latency_packets--;
        return;
    } else if (non_zero_latency_packets > 0) {
        packets.push_back(MyPacket(stream->current_queue_id, neighbor, neighbor));
        packets.back().sender = sender;
        locked_packets.push_back(&packets.back());
        processed = false;
        send_back = false;
        NPU_to_MA = false;
        process_max_count();
        non_zero_latency_packets--;
        return;
    }
}

bool MeshAllGather::ready() {
    if (stream->state == StreamState::Created ||
        stream->state == StreamState::Ready) {
        stream->changeState(StreamState::Executing);
    }
  if (packets.size() == 0 || stream_count == 0 || free_packets == 0) {
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    logger->debug("NPU {} ready() blocked: packets={}, stream_count={}, free={}", 
                  id, packets.size(), stream_count, free_packets);
    return false;
  }
  MyPacket packet = packets.front();
  auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
  logger->info("NPU {} ready() sending to {}", id, packet.preferred_dest);
    
    // Send and receive like Ring does
    sim_request snd_req;
    snd_req.srcRank = id;
    snd_req.dstRank = packet.preferred_dest;
    snd_req.tag = stream->stream_id;
    snd_req.reqType = UINT8;
    snd_req.vnet = this->stream->current_queue_id;
    
    stream->owner->front_end_sim_send(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_dest,
        stream->stream_id, &snd_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, nullptr);
    
    sim_request rcv_req;
    rcv_req.vnet = this->stream->current_queue_id;
    RecvPacketEventHandlerData* ehd = new RecvPacketEventHandlerData(
        stream, stream->owner->id, EventType::PacketReceived,
        packet.preferred_vnet, packet.stream_id);
    
    stream->owner->front_end_sim_recv(
        0, Sys::dummy_data, msg_size, UINT8, packet.preferred_src,
        stream->stream_id, &rcv_req, Sys::FrontEndSendRecvType::COLLECTIVE,
        &Sys::handleEvent, ehd);
    
    reduce();
    return true;
}

void MeshAllGather::run(EventType event, CallData* data) {
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    
    if (event == EventType::General) {
        free_packets += 1;
    logger->info("NPU {} EventType::General, free_packets now {}, calling ready()", id, free_packets);
        ready();
        iteratable();
    } else if (event == EventType::PacketReceived) {
        total_packets_received++;
    logger->info("NPU {} PacketReceived #{}: pending {} -> {}", 
                 id, total_packets_received, pending_receives_this_round, pending_receives_this_round - 1);
    // When all expected recvs of this round have arrived, schedule next round.
    if (pending_receives_this_round > 0) {
      pending_receives_this_round--;
    }
    logger->info("NPU {} after decr: pending={}, round={}/{}", 
                 id, pending_receives_this_round, current_round, max_hops);
    if (pending_receives_this_round == 0 && current_round < max_hops) {
      auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
      logger->info("NPU {} -> round {}", id, current_round + 1);
      current_round++;
      if (current_round < max_hops) {
        schedule_round();
      }
    }
    } else if (event == EventType::StreamInit) {
    logger->info("NPU {} EventType::StreamInit, starting flooding with {} neighbors (max_hops={})", id, (int)neighbors.size(), max_hops);
    // Kick off round 0: send to all neighbors
    schedule_round();
    }
}

void MeshAllGather::exit() {
    auto logger = LoggerFactory::get_logger("system::collective::MeshAllGather");
    logger->info("NPU {} exiting MeshAllGather: sent={}, received={}, received_from={}/{} NPUs", 
                 id, total_packets_sent, total_packets_received, received_from_npus.size(), total_nodes);
    
    if (!packets.empty()) {
        packets.clear();
    }
    if (!locked_packets.empty()) {
        locked_packets.clear();
    }
    
    stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}

