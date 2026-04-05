import json
import os
import shutil

# Paths
PROJECT_ROOT = "/home/haotian/waferchip/mesh2D/astra_sim_picasso"
NETWORK_DIR = os.path.join(PROJECT_ROOT, "examples/network/analytical")
SYSTEM_DIR = os.path.join(PROJECT_ROOT, "examples/system/native_collectives")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "examples/run_scripts/analytical/congestion_aware/configs")
WORKLOADS_BASE_DIR = "{PROJECT_DIR}/examples/workloads/all_reduce_scaling"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Generate Physical Topology YAMLs
# -----------------------------------------------------------------------------
def generate_mesh_yml(gpu_count, width, height, output_path):
    content = f"""topology: [ Mesh2D ]
npus_count: [ {gpu_count} ]
width: {width}
height: {height}
bandwidth: [ 400.0 ]  # GB/s
latency: [ 900.0 ]    # ns

# Generated Mesh2D Topology ({width}x{height})
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated Network: {output_path}")

def generate_fc_yml(gpu_count, output_path):
    content = f"""topology: [ FullyConnected ]
npus_count: [ {gpu_count} ]
bandwidth: [ 400.0 ]  # GB/s
latency: [ 900.0 ]    # ns

# Generated FullyConnected Topology ({gpu_count} GPUs)
"""
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Generated Network: {output_path}")

# -----------------------------------------------------------------------------
# 2. Generate System JSONs (Ring)
# -----------------------------------------------------------------------------
def generate_ring_system_json(gpu_count, output_path):
    content = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 2,
        "preferred-dataset-splits": gpu_count,
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["ring"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": 3350,
        "boost-mode": 0,
        "roofline-enabled": 1,
        "peak-perf": 989
    }
    with open(output_path, "w") as f:
        json.dump(content, f, indent=4)
    print(f"Generated System: {output_path}")

# -----------------------------------------------------------------------------
# 3. Generate Run Configs
# -----------------------------------------------------------------------------
def generate_run_config(gpu_count, physical_topo_name, network_file, system_file):
    # Naming convention: comm_all_reduce_{N}gpus_{phys}_{logic}.json
    # logic is always 'ring' here.
    # phys: 'fc' or 'mesh2d'
    
    config_name = f"comm_all_reduce_{gpu_count}gpus_{physical_topo_name}_ring"
    workload_name = f"all_reduce_{gpu_count}gpus"
    
    config_content = {
        "workload_dir": f"{WORKLOADS_BASE_DIR}/{workload_name}",
        "workload_base": workload_name,
        "comm_group": f"{WORKLOADS_BASE_DIR}/{workload_name}/{workload_name}.json",
        "system": f"{{EXAMPLE_DIR}}/system/native_collectives/{system_file}",
        "network": f"{{EXAMPLE_DIR}}/network/analytical/{network_file}",
        "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
        "output_dir": f"{{PROJECT_DIR}}/output/scaling_test/{config_name}",
        "log_file": f"{config_name}.log"
    }
    
    output_path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    with open(output_path, "w") as f:
        json.dump(config_content, f, indent=4)
    print(f"Generated Config: {output_path}")

def main():
    ensure_dir(NETWORK_DIR)
    ensure_dir(SYSTEM_DIR)
    ensure_dir(CONFIGS_DIR)

    # Clean up old scaling configs
    for f in os.listdir(CONFIGS_DIR):
        if f.startswith("scaling_all_reduce_"):
            os.remove(os.path.join(CONFIGS_DIR, f))
            print(f"Removed old config: {f}")

    scaling_steps = [
        {"count": 2, "mesh_dims": (2, 1)},
        {"count": 4, "mesh_dims": (2, 2)},
        {"count": 8, "mesh_dims": (4, 2)}, # 4 cols, 2 rows
    ]

    for step in scaling_steps:
        cnt = step["count"]
        w, h = step["mesh_dims"]
        
        # 1. Network Files
        mesh_file = f"Mesh2D_{cnt}gpus_{w}x{h}_H100.yml"
        fc_file = f"FullyConn_{cnt}gpus_H100.yml"
        
        generate_mesh_yml(cnt, w, h, os.path.join(NETWORK_DIR, mesh_file))
        generate_fc_yml(cnt, os.path.join(NETWORK_DIR, fc_file))
        
        # 2. System Files
        system_file = f"ring{cnt}_H100.json"
        generate_ring_system_json(cnt, os.path.join(SYSTEM_DIR, system_file))
        
        # 3. Run Configs
        generate_run_config(cnt, "mesh2d", mesh_file, system_file)
        generate_run_config(cnt, "fc", fc_file, system_file)

if __name__ == "__main__":
    main()
