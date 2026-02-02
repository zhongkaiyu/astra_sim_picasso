import argparse
import os
import sys

# Ensure repo-local chakra package is importable without extra env setup.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
GRAPH_FRONTEND_ROOT = os.path.join(REPO_ROOT, "extern", "graph_frontend")
if GRAPH_FRONTEND_ROOT not in sys.path:
    sys.path.insert(0, GRAPH_FRONTEND_ROOT)

from chakra.src.third_party.utils.protolib import encodeMessage as encode_message
from chakra.schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
    BoolList,
    GlobalMetadata,
    AttributeProto as ChakraAttr,
    COMM_COLL_NODE,
    ALL_REDUCE,
    REDUCE,
    ALL_GATHER,
    GATHER,
    SCATTER,
    BROADCAST,
    ALL_TO_ALL,
    REDUCE_SCATTER,
    REDUCE_SCATTER_BLOCK,
    BARRIER,
)


def _comm_type_map():
    return {
        "all_reduce": ALL_REDUCE,
        "reduce": REDUCE,
        "all_gather": ALL_GATHER,
        "gather": GATHER,
        "scatter": SCATTER,
        "broadcast": BROADCAST,
        "all_to_all": ALL_TO_ALL,
        "reduce_scatter": REDUCE_SCATTER,
        "reduce_scatter_block": REDUCE_SCATTER_BLOCK,
        "barrier": BARRIER,
    }


def _write_trace(output_path: str, comm_type: int, comm_name: str, coll_size: int) -> None:
    with open(output_path, "wb") as et:
        encode_message(et, GlobalMetadata(version="0.0.4"))

        node = ChakraNode()
        node.id = 1
        node.name = comm_name
        node.type = COMM_COLL_NODE

        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=comm_type))
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--npus", type=int, default=16, help="number of NPUs")
    parser.add_argument(
        "--comm_size",
        type=int,
        default=1_048_576,
        help="collective payload size (bytes)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "comm_primitive_traces_16gpu"),
        help="output directory for traces",
    )
    args = parser.parse_args()

    comm_map = _comm_type_map()
    for comm_name, comm_type in comm_map.items():
        comm_dir = os.path.join(args.out_dir, comm_name)
        os.makedirs(comm_dir, exist_ok=True)
        _write_comm_group(os.path.join(comm_dir, "chakra_trace.json"), args.npus)
        for npu_id in range(args.npus):
            output_filename = os.path.join(comm_dir, f"chakra_trace.{npu_id}.et")
            _write_trace(output_filename, comm_type, comm_name, args.comm_size)


if __name__ == "__main__":
    main()
