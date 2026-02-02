#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def _setup_python_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    chakra_root = repo_root / "extern" / "graph_frontend" / "chakra"
    sys.path.insert(0, str(chakra_root / "src"))
    sys.path.insert(0, str(chakra_root / "schema" / "protobuf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Chakra ET to a valid JSON array."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Chakra ET file (*.et or *.et.gz).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    _setup_python_path()
    from google.protobuf.json_format import MessageToJson
    from third_party.utils.protolib import decodeMessage as decode_message
    from third_party.utils.protolib import openFileRd as open_file_rd
    import et_def_pb2

    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    execution_trace = open_file_rd(str(input_path))
    with open(output_path, "w") as f:
        f.write("[\n")
        first = True

        global_metadata = et_def_pb2.GlobalMetadata()
        decode_message(execution_trace, global_metadata)
        f.write(MessageToJson(global_metadata).rstrip())
        first = False

        node = et_def_pb2.Node()
        while decode_message(execution_trace, node):
            if not first:
                f.write(",\n")
            f.write(MessageToJson(node).rstrip())
            first = False
            node.Clear()

        f.write("\n]\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




