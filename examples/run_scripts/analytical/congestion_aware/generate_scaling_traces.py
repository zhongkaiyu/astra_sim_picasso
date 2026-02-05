import argparse
import os
import sys

# Ensure repo-local chakra package is importable without extra env setup.
# Adjust paths based on where this script is located: examples/run_scripts/analytical/congestion_aware/
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# ../../../../
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
GRAPH_FRONTEND_ROOT = os.path.join(REPO_ROOT, "extern", "graph_frontend")
if GRAPH_FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, GRAPH_FRONTEND_ROOT)

from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
)

def _write_trace(output_path: str, comm_type: int, comm_name: str, coll_size: int) -> None:
    with open(output_path, "wb") as et:
        encode_message(et, GlobalMetadata(version="0.0.4"))

        node = ChakraNode()
        node.id = 1
        node.name = comm_name
        node.type = COMM_COLL_NODE

        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=comm_type))
        # comm_size in bytes
        node.attr.append(ChakraAttr(name="comm_size", int64_val=coll_size))

        encode_message(et, node)

def _write_comm_group(output_path: str, npus: int) -> None:
    # Minimal comm group: one collective node (id=1) includes all NPUs.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("{")
        f.write('"1": [')
        f.write(", ".join(str(i) for i in range(npus)))
        f.write("]}")

def main() -> None:
    # Sizes: 2, 4, 8
    # Default size: 64MB
    comm_size = 64 * 1024 * 1024 
    
    # Output dir: examples/workloads/all_reduce_scaling
    workloads_dir = os.path.join(REPO_ROOT, "examples", "workloads", "all_reduce_scaling")
    
    counts = [2, 4, 8]
    
    for npus in counts:
        print(f"Generating AllReduce trace for {npus} NPUs...")
        out_dir = os.path.join(workloads_dir, f"all_reduce_{npus}gpus")
        os.makedirs(out_dir, exist_ok=True)
        
        # Write comm group (JSON)
        # Note: Astra-sim usually looks for a .json file that defines the comm group.
        # The filename often matches the ET filename pattern or is passed explicitly.
        # We will name it `all_reduce_{npus}gpus.json` to match the folder/trace name style usually expected.
        comm_group_path = os.path.join(out_dir, f"all_reduce_{npus}gpus.json")
        _write_comm_group(comm_group_path, npus)
        
        # Write traces (ET)
        for npu_id in range(npus):
            # Format: all_reduce_{npus}gpus.{npu_id}.et
            trace_path = os.path.join(out_dir, f"all_reduce_{npus}gpus.{npu_id}.et")
            _write_trace(trace_path, ALL_REDUCE, "all_reduce", comm_size)
            
    print(f"Done! Traces generated in {workloads_dir}")

if __name__ == "__main__":
    main()
