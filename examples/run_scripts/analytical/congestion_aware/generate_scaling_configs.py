import json
import os

# Configuration
CONFIGS_DIR = "/home/haotian/waferchip/mesh2D/astra_sim_picasso/examples/run_scripts/analytical/congestion_aware/configs"
WORKLOADS_BASE_DIR = "{PROJECT_DIR}/examples/workloads/all_reduce_scaling"

# Based on file listing, we map available files or use close substitutes
NETWORKS = {
    2: {
        # No direct 2-GPU file found, reusing 4-GPU topology but will only use 2 nodes in simulation
        "mesh2d": "{EXAMPLE_DIR}/network/analytical/Mesh2D_4gpus_2x2_H100.yml", 
        "fc": "{EXAMPLE_DIR}/network/analytical/AllToAll_4gpus_H100.yml"
    },
    4: {
        "mesh2d": "{EXAMPLE_DIR}/network/analytical/Mesh2D_4gpus_2x2_H100.yml",
        "fc": "{EXAMPLE_DIR}/network/analytical/AllToAll_4gpus_H100.yml"
    },
    8: {
        # No direct 8-GPU file found, using NVSwitch_8gpus for FC, and 16-GPU mesh for Mesh (subset)
        "mesh2d": "{EXAMPLE_DIR}/network/analytical/Mesh2D_16gpus_4x4_H100.yml",
        "fc": "{EXAMPLE_DIR}/network/analytical/NVSwitch_8gpus_megatron.yml"
    }
}

# System templates
# We only saw DP4_H100.json. We should probably create DP2 and DP8 or reuse DP4/16 as template.
# For this script, we will point to standard names, and if they don't exist, we will create them in the next step.
SYSTEMS = {
    2: "{EXAMPLE_DIR}/system/native_collectives/DP4_H100.json", # Reusing DP4 for now
    4: "{EXAMPLE_DIR}/system/native_collectives/DP4_H100.json",
    8: "{EXAMPLE_DIR}/system/native_collectives/DP16_H100_AllToAll.json" # Reusing DP16 for now
}

def generate_config(gpu_count, topology, network_path, system_path):
    workload_name = f"all_reduce_{gpu_count}gpus"
    config_name = f"scaling_all_reduce_{gpu_count}gpus_{topology}"
    
    config_content = {
        "workload_dir": f"{WORKLOADS_BASE_DIR}/{workload_name}",
        "workload_base": workload_name,
        "comm_group": f"{WORKLOADS_BASE_DIR}/{workload_name}/{workload_name}.json",
        "system": system_path,
        "network": network_path,
        "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
        "output_dir": f"{{PROJECT_DIR}}/output/scaling_test/{config_name}",
        "log_file": f"{config_name}.log"
    }
    
    output_path = os.path.join(CONFIGS_DIR, f"{config_name}.json")
    with open(output_path, "w") as f:
        json.dump(config_content, f, indent=4)
    print(f"Generated: {output_path}")

def main():
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    
    for gpu_count in [2, 4, 8]:
        generate_config(
            gpu_count, 
            "mesh2d", 
            NETWORKS[gpu_count]["mesh2d"], 
            SYSTEMS[gpu_count]
        )
        generate_config(
            gpu_count, 
            "fc", 
            NETWORKS[gpu_count]["fc"], 
            SYSTEMS[gpu_count]
        )

if __name__ == "__main__":
    main()
