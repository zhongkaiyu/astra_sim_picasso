/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __MESH_ALL_GATHER_HH__
#define __MESH_ALL_GATHER_HH__

#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/astraccl/Algorithm.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/Mesh2DTopology.hh"

#include <set>
#include <map>

namespace AstraSim {

/**
 * MeshAllGather implements AllGather on a 2D mesh topology using flooding algorithm.
 * 
 * Algorithm Description:
 * - Each NPU starts by sending its data to all neighbors (East, West, North, South)
 * - When a packet is received from another NPU, forward it to all neighbors that 
 *   haven't received it yet
 * - Continue until all NPUs have received data from all other NPUs
 * - Uses Manhattan distance to determine when all data has been collected
 * 
 * Time Complexity: O(max(width, height)) for 2D mesh
 * 
 * Example on 4x4 mesh:
 * - Max distance from any NPU to any other: 6 hops
 * - Each NPU maintains a set of received NPU IDs
 * - When set.size() == total_nodes, all data is gathered
 */
class MeshAllGather : public Algorithm {
  public:
    MeshAllGather(ComType type,
                  int id,
                  Mesh2DTopology* mesh_topology,
                  uint64_t data_size);
    
    // Override Algorithm's virtual method
    void run(EventType event, CallData* data) override;
    
    void process_stream_count();
    void release_packets();
    void process_max_count();
    void reduce();
    bool iteratable();
    virtual int get_non_zero_latency_packets();
    void insert_packet(Callable* sender);
    bool ready();
    void exit();
    
    /**
     * Forward a received packet to neighbors that haven't seen it yet.
     * 
     * @param received_from_npu the NPU ID that sent this packet
     */
    void forward_to_neighbors(int received_from_npu);

  private:
    Mesh2DTopology::Direction dimension;
    MemBus::Transmition transmition;
    int zero_latency_packets;
    int non_zero_latency_packets;
    int id;
    int curr_receiver;
    int curr_sender;
    int total_nodes;
    int stream_count;
    int max_count;
    int remained_packets_per_max_count;
    int remained_packets_per_message;
    int parallel_reduce;
    InjectionPolicy injection_policy;
    std::list<MyPacket> packets;
    bool toggle;
    long free_packets;
    long total_packets_sent;
    long total_packets_received;
    uint64_t msg_size;
    std::list<MyPacket*> locked_packets;
    bool processed;
    bool send_back;
    bool NPU_to_MA;

    // Mesh-specific members
    std::set<int> received_from_npus;  // Track which NPUs we've received data from
    Mesh2DTopology* mesh_topo;
    int max_hops;  // Maximum hops needed (Manhattan distance across mesh diagonal)

    // Flooding state
    std::vector<int> neighbors;  // Adjacent NPUs
    int num_neighbors;
    int current_round;  // 0..max_hops-1
    int pending_receives_this_round;  // how many recvs to await this round

    // Schedule sends/recvs to all neighbors for the current round
    void schedule_round();
};

}  // namespace AstraSim

#endif /* __MESH_ALL_GATHER_HH__ */

