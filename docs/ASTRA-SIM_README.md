# ASTRA-sim Documentation

A comprehensive guide to compiling, understanding, and extending ASTRA-sim for distributed deep learning simulation.

---

## Table of Contents
1. [Quick Start: Compilation](#1-quick-start-compilation)
2. [Mesh2D Physical Backend](#2-mesh2d-physical-backend)
3. [Adding New Logical Communication Logic](#3-adding-new-logical-communication-logic)
4. [Using Example Scripts](#4-using-example-scripts)
5. [Configuration Reference](#5-configuration-reference)

---

## 1. Quick Start: Compilation


### Building ASTRA-sim with Analytical Backend

The analytical backend is the recommended way to run simulations. It provides both congestion-aware and congestion-unaware modes.

```bash
# Navigate to project root
cd /path/to/astra-sim

# Build all targets (congestion_aware + congestion_unaware)
./build/astra_analytical/build.sh

# Or build specific target
./build/astra_analytical/build.sh -t congestion_aware
./build/astra_analytical/build.sh -t congestion_unaware
```

### Build Outputs

After successful compilation, binaries are located at:

```
build/astra_analytical/build/bin/
├── AstraSim_Analytical_Congestion_Aware    # Congestion-aware simulator
└── AstraSim_Analytical_Congestion_Unaware  # Congestion-unaware simulator
```

---

## 2. Mesh2D Architecture Overview

### Three-Layer Architecture

ASTRA-sim separates network simulation into three distinct layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 3: COLLECTIVE ALGORITHMS                       │
│  What: Implements communication patterns (AllGather, AllReduce, etc.)       │
│  How:  Decides WHICH NPUs to send/receive from and in WHAT ORDER            │
│  Files: MeshAllGather.cc, RingAllReduce.cc, etc.                            │
│  Uses:  Logical Topology layer for neighbor/distance queries                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Queries: "Who are my neighbors?"
                                    │          "How far is NPU X?"
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: LOGICAL TOPOLOGY                            │
│  What: Provides topology-aware utilities for algorithms                     │
│  How:  Coordinate mapping, neighbor lookup, distance calculation            │
│  Files: Mesh2DTopology.cc, RingTopology.cc, etc.                            │
│  Note:  NO actual packet transmission - just topology queries               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Sends packets via Network API
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: PHYSICAL NETWORK                            │
│  What: Simulates actual packet transmission with timing                     │
│  How:  XY routing, congestion modeling, bandwidth/latency simulation        │
│  Files: Mesh2D.cpp (in network_backend)                                     │
│  Note:  Handles HOW packets physically traverse the network                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: Physical Network (Mesh2D Backend)

### Purpose
Simulates the **actual physical network** - how packets travel between NPUs with real timing, congestion, and routing.

### Topology Structure

A Mesh2D with `width=6` and `height=4` creates a 24-NPU grid:

```
     0 --- 1 --- 2 --- 3 --- 4 --- 5
     |     |     |     |     |     |
     6 --- 7 --- 8 --- 9 --- 10--- 11
     |     |     |     |     |     |
     12--- 13--- 14--- 15--- 16--- 17
     |     |     |     |     |     |
     18--- 19--- 20--- 21--- 22--- 23
```

### What This Layer Does
- **Creates physical links** between adjacent NPUs (bandwidth, latency per link)
- **Routes packets** using XY routing algorithm
- **Models congestion** when multiple packets compete for same link
- **Calculates transmission time** based on packet size, bandwidth, hop count

### Configuration (YAML)

```yaml
# examples/network/analytical/Mesh2D_24npus_6x4.yml
topology: [ Mesh2D ]
npus_count: [ 24 ]
width: 6            # Number of columns
height: 4           # Number of rows
bandwidth: [ 100.0 ]  # GB/s per link
latency: [ 1000.0 ]   # Nanoseconds per hop
```

### Files Added

| File | Purpose |
|------|---------|
| `extern/network_backend/analytical/include/.../Mesh2D.h` | Header: topology interface, routing function signatures |
| `extern/network_backend/analytical/congestion_aware/basic-topology/Mesh2D.cpp` | Implementation: link creation, XY routing, congestion handling |

### Integration Point

In `extern/network_backend/analytical/congestion_aware/topology/Helper.cpp`:

```cpp
case TopologyBuildingBlock::Mesh2D:
    return std::make_shared<Mesh2D>(mesh_width, mesh_height, bandwidth, latency);
```

---

## 4. Layer 2: Logical Topology (Mesh2DTopology)

### Purpose
Provides **topology-aware query utilities** for collective algorithms. Does NOT send packets - only answers questions about the topology.

### What This Layer Does
- **Coordinate conversion**: NPU ID ↔ (x, y) coordinates
- **Neighbor lookup**: "Who is to my East/West/North/South?"
- **Distance calculation**: Manhattan distance between any two NPUs
- **Boundary detection**: "Am I on an edge? Which directions have no neighbor?"

### Key Difference from Physical Layer

| Aspect | Physical Layer (Mesh2D.cpp) | Logical Layer (Mesh2DTopology.cc) |
|--------|----------------------------|-----------------------------------|
| Purpose | Transmit packets with timing | Answer topology queries |
| Output | Simulated packet delivery | Neighbor IDs, distances, coordinates |
| Congestion | Yes, models it | No, just topology structure |
| Used by | Network backend | Collective algorithms |

### Files Added

| File | Purpose |
|------|---------|
| `astra-sim/system/astraccl/native_collectives/logical_topology/Mesh2DTopology.hh` | Header: query interface |
| `astra-sim/system/astraccl/native_collectives/logical_topology/Mesh2DTopology.cc` | Implementation of utilities |

---

## 5. Layer 3: Collective Algorithms

### Purpose
Implements the **communication pattern logic** - which NPUs send to which, in what order, for operations like AllGather, AllReduce, etc.

### What This Layer Does
- **Schedules sends/receives** based on algorithm logic
- **Uses Logical Topology** to query neighbors and distances
- **Triggers Physical Network** to actually transmit packets

### Files Added (Placeholder Implementation)

| File | Purpose |
|------|---------|
| `astra-sim/system/astraccl/native_collectives/collective_algorithm/MeshAllGather.hh` | Header for mesh-aware AllGather |
| `astra-sim/system/astraccl/native_collectives/collective_algorithm/MeshAllGather.cc` | Flooding-based AllGather implementation |

These are placeholder files. The actual algorithm needs proper implementation.

### How to Add a New Collective Algorithm

**Step 1**: Add enum in `astra-sim/system/Common.hh`:

```cpp
enum class CollectiveImplType {
    Ring = 0,
    // ... existing ...
    Mesh2D,           // ← Added for mesh algorithms
};
```

**Step 2**: Parse string in `astra-sim/system/Sys.cc`:

```cpp
} else if (collective_impl_str == "mesh2d") {
    return new CollectiveImpl(CollectiveImplType::Mesh2D);
}
```

**Step 3**: Create algorithm files:

```
astra-sim/system/astraccl/native_collectives/collective_algorithm/
├── MeshAllGather.hh
└── MeshAllGather.cc
```

**Step 4**: Add dispatch in `Sys.cc`:

```cpp
} else if (collective_impl->type == CollectiveImplType::Mesh2D) {
    CollectivePhase vn(this, queue_id,
        new MeshAllGather(collective_type, id,
                          (Mesh2DTopology*)topology, data_size));
    return vn;
}
```

---

## 6. Configuration Files

### Network Config (Physical Layer)

| File | Description |
|------|-------------|
| `examples/network/analytical/Mesh2D_16npus.yml` | 4×4 mesh |
| `examples/network/analytical/Mesh2D_24npus_6x4.yml` | 6×4 mesh |

### System Config (Algorithm Selection)

| File | Description |
|------|-------------|
| `examples/system/native_collectives/Mesh2D_AllGather.json` | Uses `mesh2d` algorithm |

```json
{
    "all-gather-implementation": ["mesh2d"],
    "all-reduce-implementation": ["ring"],
    ...
}
```

---

## 7. Communicator Groups

Communicator groups define logical NPU groupings for collective operations (TP, DP, PP groups).

**Format**: JSON mapping group IDs to NPU lists

```json
{
    "1": [0, 1, 2, 3],      
    "2": [4, 5, 6, 7],      
    "3": [0, 4, 8, 12],     
    "4": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
}
```

When generating Chakra traces, this is output as a `.json` file and specified in the run script.

---

## 4. Using Example Scripts

### Directory Structure

```
examples/
├── run_scripts/
│   ├── analytical/
│   │   ├── congestion_aware/       # Congestion-aware examples
│   │   │   ├── Mesh2D_allgather_16npus.sh
│   │   │   ├── Mesh2D_allgather_24npus_6x4.sh
│   │   │   ├── Ring_allgather_16npus.sh
│   │   │   └── run_analytical_with_custom_collective.sh
│   │   └── congestion_unaware/     # Congestion-unaware examples
│   ├── htsim/                      # HTSim backend examples
│   └── ns3/                        # NS-3 backend examples
├── network/analytical/             # Network config files
├── system/                         # System config files
│   ├── native_collectives/         # Built-in collective configs
│   └── custom_collectives/         # Custom collective definitions
├── workload/                       # Workload traces
└── remote_memory/                  # Memory configuration
```

### Running an Example

**Basic Example: 24-NPU Mesh AllGather**

```bash
cd /path/to/astra-sim
./examples/run_scripts/analytical/congestion_aware/Mesh2D_allgather_24npus_6x4.sh
```

**Script Breakdown:**

```bash
#!/bin/bash
set -e

# Paths to configuration files
ASTRA_SIM="build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="path/to/workload/traces"
SYSTEM="examples/system/native_collectives/Ring_4chunks.json"
NETWORK="examples/network/analytical/Mesh2D_24npus_6x4.yml"
REMOTE_MEMORY="examples/remote_memory/analytical/no_memory_expansion.json"
COMM_GROUP="path/to/communicator_group.json"

# Run simulation
"${ASTRA_SIM}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM}" \
    --remote-memory-configuration="${REMOTE_MEMORY}" \
    --network-configuration="${NETWORK}" \
    --comm-group-configuration="${COMM_GROUP}"
```

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--workload-configuration` | Path to workload ET files (without `.et` extension) | Yes |
| `--system-configuration` | Path to system JSON config | Yes |
| `--network-configuration` | Path to network YAML config | Yes |
| `--remote-memory-configuration` | Path to memory JSON config | Yes |
| `--comm-group-configuration` | Path to communicator groups JSON | Optional |
| `--logging-configuration` | Path to logging config | Optional |
| `--comm-scale` | Communication scaling factor | Optional |
| `--injection-scale` | Injection scaling factor | Optional |


---

## 5. Configuration Reference

### Network Configuration (YAML)

| Parameter | Type | Description |
|-----------|------|-------------|
| `topology` | list[str] | Topology type: `Ring`, `FullyConnected`, `Switch`, `Mesh2D` |
| `npus_count` | list[int] | NPUs per dimension |
| `bandwidth` | list[float] | Bandwidth in GB/s per dimension |
| `latency` | list[float] | Latency in nanoseconds per dimension |
| `width` | int | (Mesh2D only) Number of columns |
| `height` | int | (Mesh2D only) Number of rows |

**Example Network Configs:**


```yaml
# 4x4 Mesh (16 NPUs)
topology: [ Mesh2D ]
npus_count: [ 16 ]
width: 4
height: 4
bandwidth: [ 100.0 ]
latency: [ 1000.0 ]
```

### System Configuration (JSON)

| Parameter | Type | Description |
|-----------|------|-------------|
| `scheduling-policy` | str | `LIFO`, `FIFO`, `EXPLICIT` |
| `endpoint-delay` | int | Delay at endpoints (ns) |
| `active-chunks-per-dimension` | int | Concurrent chunks per dim |
| `preferred-dataset-splits` | int | Data chunking factor |
| `all-reduce-implementation` | list[str] | Algorithm per dimension |
| `all-gather-implementation` | list[str] | Algorithm per dimension |
| `reduce-scatter-implementation` | list[str] | Algorithm per dimension |
| `all-to-all-implementation` | list[str] | Algorithm per dimension |
| `collective-optimization` | str | `Baseline`, `localBWAware` |
| `local-mem-bw` | int | Local memory bandwidth (GB/s) |
| `boost-mode` | int | Enable boost mode (0/1) |
| `roofline-enabled` | int | Enable roofline model (0/1) |
| `peak-perf` | int | Peak FLOPS (TFLOPS) |



*Last updated: December 2024*

