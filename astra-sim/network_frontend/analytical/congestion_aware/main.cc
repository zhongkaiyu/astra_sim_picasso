/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "common/CmdLineParser.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

int main(int argc, char* argv[]) {
    std::cerr << "\n";
    std::cerr << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                    ASTRA-SIM: MAIN ENTRY POINT                            ║\n";
    std::cerr << "║               Analytical Network Backend - Congestion Aware               ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cerr << "[MAIN] Starting ASTRA-SIM simulation...\n";
    std::cerr << "[MAIN] Process ID: " << getpid() << "\n";
    std::cerr << "[MAIN] Timestamp: " << __DATE__ << " " << __TIME__ << "\n\n";

    // Parse command line arguments
    std::cerr << "[MAIN-ARGS] Parsing command line arguments...\n";
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);
    std::cerr << "[MAIN-ARGS] Arguments parsed ✓\n\n";

    // Get command line arguments
    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration =
        cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    const auto logging_folder =
        cmd_line_parser.get<std::string>("logging-folder");
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");

    std::cerr << "[MAIN-CONFIG] Configuration files:\n";
    std::cerr << "  ├─ Workload: " << workload_configuration << "\n";
    std::cerr << "  ├─ System: " << system_configuration << "\n";
    std::cerr << "  ├─ Network: " << network_configuration << "\n";
    std::cerr << "  ├─ Remote Memory: " << remote_memory_configuration << "\n";
    std::cerr << "  ├─ Comm Groups: " << comm_group_configuration << "\n";
    std::cerr << "  └─ Logging: " << logging_configuration << "\n\n";

    AstraSim::LoggerFactory::init(logging_configuration, logging_folder);

    // Instantiate event queue
    std::cerr << "[MAIN-QUEUE] Creating event queue (Discrete Event Simulator)...\n";
    const auto event_queue = std::make_shared<EventQueue>();
    Topology::set_event_queue(event_queue);
    std::cerr << "[MAIN-QUEUE] Event queue created ✓\n\n";

    // Generate topology
    std::cerr << "[MAIN-TOPOLOGY] Parsing and constructing network topology...\n";
    const auto network_parser = NetworkParser(network_configuration);
    std::cerr << "[MAIN-TOPOLOGY] NetworkParser created from: " << network_configuration << "\n";
    
    const auto topology = construct_topology(network_parser);
    std::cerr << "[MAIN-TOPOLOGY] Topology constructed ✓\n";

    // Get topology information
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();

    std::cerr << "[MAIN-TOPOLOGY] Topology info:\n";
    std::cerr << "  ├─ Total NPUs: " << npus_count << "\n";
    std::cerr << "  ├─ Dimensions: " << dims_count << "\n";
    std::cerr << "  └─ NPUs per dimension: ";
    for (int i = 0; i < npus_count_per_dim.size(); i++) {
        std::cerr << npus_count_per_dim[i];
        if (i < npus_count_per_dim.size() - 1) std::cerr << " × ";
    }
    std::cerr << "\n\n";

    // Set up Network API
    std::cerr << "[MAIN-NETWORK] Setting up Network API...\n";
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    std::cerr << "[MAIN-NETWORK] Network API configured ✓\n\n";

    // Create ASTRA-sim related resources
    std::cerr << "[MAIN-RESOURCES] Allocating ASTRA-sim resources...\n";
    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = std::vector<Sys*>();
    std::cerr << "[MAIN-RESOURCES] Memory API created ✓\n";

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    // Create Sys objects (one per NPU)
    std::cerr << "\n╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                    PHASE 1: SYSTEM INITIALIZATION                         ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cerr << "[SYS-INIT] Creating " << npus_count << " Sys objects (one per NPU)...\n\n";

    for (int i = 0; i < npus_count; i++) {
        std::cerr << "[SYS-INIT] Creating Sys[" << i << "]...\n";
        
        // create network and system
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        std::cerr << "[SYS-INIT]   ├─ Network API created\n";
        
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, injection_scale,
                    comm_scale, rendezvous_protocol);

        std::cerr << "[SYS-INIT]   ├─ Sys object created\n";
        std::cerr << "[SYS-INIT]   └─ Workload loaded ✓\n";

        // push back network and system
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    std::cerr << "\n[SYS-INIT] All " << npus_count << " Sys objects created ✓\n\n";

    // Initiate ASTRA-sim simulation
    std::cerr << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                    PHASE 2: FIRE INITIAL EVENTS                           ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cerr << "[SIM-START] Firing initial events for all NPUs...\n";
    for (int i = 0; i < npus_count; i++) {
        std::cerr << "[SIM-START] Calling workload[" << i << "]->fire()\n";
        systems[i]->workload->fire();
    }
    std::cerr << "\n[SIM-START] All initial events fired ✓\n\n";

    // run simulation
    std::cerr << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║              PHASE 3: MAIN SIMULATION LOOP (Discrete Events)              ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cerr << "[SIM-LOOP] Starting main simulation loop...\n";
    std::cerr << "[SIM-LOOP] Processing events from queue until completion\n\n";

    int iteration = 0;
    while (!event_queue->finished()) {
        iteration++;
        if (iteration % 100 == 0) {  // Print every 100 iterations to avoid spam
            //std::cerr << "[SIM-LOOP] Iteration " << iteration 
            //          << ", Simulation time: " << Sys::boostedTick() << "\n";
        }
        event_queue->proceed();
    }

    std::cerr << "\n[SIM-LOOP] Total iterations: " << iteration << "\n";
    std::cerr << "[SIM-LOOP] Event queue finished - Simulation COMPLETE ✓\n\n";

    // Cleanup
    std::cerr << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                         CLEANUP AND SHUTDOWN                              ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    std::cerr << "[SHUTDOWN] Deleting Sys objects...\n";
    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    std::cerr << "[SHUTDOWN] Sys objects deleted ✓\n";

    // terminate simulation
    std::cerr << "[SHUTDOWN] Shutting down logger...\n";
    AstraSim::LoggerFactory::shutdown();
    std::cerr << "[SHUTDOWN] Logger shutdown ✓\n\n";

    std::cerr << "╔═══════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                      ASTRA-SIM EXECUTION COMPLETE                         ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
