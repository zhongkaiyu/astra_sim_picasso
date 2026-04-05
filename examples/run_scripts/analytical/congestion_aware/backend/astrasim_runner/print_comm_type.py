#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path


def _setup_python_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    chakra_root = repo_root / "extern" / "graph_frontend" / "chakra"
    sys.path.insert(0, str(chakra_root / "src"))
    sys.path.insert(0, str(chakra_root / "schema" / "protobuf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print nodes with comm_type from a Chakra ET file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Chakra ET file (*.et or *.et.gz).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print comm_type summary.",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only print nodes whose name contains this substring.",
    )
    return parser.parse_args()


def _get_attr_int64(node, attr_name: str):
    for attr in node.attr:
        if attr.name == attr_name:
            return attr.int64_val
    return None


def main() -> int:
    _setup_python_path()
    from third_party.utils.protolib import decodeMessage as decode_message
    from third_party.utils.protolib import openFileRd as open_file_rd
    import et_def_pb2

    args = _parse_args()
    et_path = Path(args.input)
    if not et_path.exists():
        print(f"Input file not found: {et_path}")
        return 1

    counter = Counter()
    printed = 0

    execution_trace = open_file_rd(str(et_path))
    global_metadata = et_def_pb2.GlobalMetadata()
    decode_message(execution_trace, global_metadata)

    node = et_def_pb2.Node()
    while decode_message(execution_trace, node):
        comm_type_val = _get_attr_int64(node, "comm_type")
        if comm_type_val is None:
            node.Clear()
            continue

        try:
            comm_type_name = et_def_pb2.CollectiveCommType.Name(comm_type_val)
        except ValueError:
            comm_type_name = f"UNKNOWN({comm_type_val})"

        counter[comm_type_name] += 1

        if not args.summary_only:
            if args.name_contains and args.name_contains not in node.name:
                node.Clear()
                continue
            node_type_name = et_def_pb2.NodeType.Name(node.type)
            comm_size = _get_attr_int64(node, "comm_size")
            size_str = f"{comm_size} bytes" if comm_size is not None else "N/A"
            print(
                f"id={node.id} type={node_type_name} comm_type={comm_type_name} "
                f"comm_size={size_str} name={node.name}"
            )
            printed += 1

        node.Clear()

    if counter:
        print("\n[COMM_TYPE SUMMARY]")
        for comm_type_name, count in counter.most_common():
            print(f"{comm_type_name}: {count}")
    else:
        print("No nodes with comm_type found.")

    if not args.summary_only:
        print(f"\nPrinted {printed} nodes with comm_type.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

