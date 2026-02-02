#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chakra ET File Analyzer
Analyze Chakra Execution Trace (.et) files

Usage:
    python analyze_et_file.py --input attention_tp8.0.et
    python analyze_et_file.py --input attention_tp8.0.et --output report.txt
    python analyze_et_file.py --input attention_tp8.0.et --stats --graph
"""

import argparse
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Add Chakra path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHAKRA_PATH = os.path.join(PROJECT_ROOT, "extern", "graph_frontend", "chakra")
sys.path.insert(0, CHAKRA_PATH)

try:
    from src.third_party.utils.protolib import decodeMessage as decode_message
    from schema.protobuf.et_def_pb2 import (
        GlobalMetadata,
        Node as ChakraNode,
        NodeType,
        CollectiveCommType,
        INVALID_NODE,
        METADATA_NODE,
        MEM_LOAD_NODE,
        MEM_STORE_NODE,
        COMP_NODE,
        COMM_SEND_NODE,
        COMM_RECV_NODE,
        COMM_COLL_NODE,
        ALL_REDUCE,
        REDUCE,
        ALL_GATHER,
        GATHER,
        SCATTER,
        BROADCAST,
        ALL_TO_ALL,
        REDUCE_SCATTER,
    )
except ImportError as e:
    print(f"Error: Cannot import Chakra protobuf modules.")
    print(f"Details: {e}")
    print(f"Please ensure Chakra is properly installed.")
    sys.exit(1)


class ETAnalyzer:
    """Chakra ET File Analyzer"""
    
    NODE_TYPE_NAMES = {
        INVALID_NODE: "INVALID",
        METADATA_NODE: "METADATA",
        MEM_LOAD_NODE: "MEM_LOAD",
        MEM_STORE_NODE: "MEM_STORE",
        COMP_NODE: "COMP",
        COMM_SEND_NODE: "COMM_SEND",
        COMM_RECV_NODE: "COMM_RECV",
        COMM_COLL_NODE: "COMM_COLL",
    }
    
    COMM_TYPE_NAMES = {
        ALL_REDUCE: "AllReduce",
        REDUCE: "Reduce",
        ALL_GATHER: "AllGather",
        GATHER: "Gather",
        SCATTER: "Scatter",
        BROADCAST: "Broadcast",
        ALL_TO_ALL: "AllToAll",
        REDUCE_SCATTER: "ReduceScatter",
    }
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.metadata = None
        self.nodes = []
        self.stats = {
            'total_nodes': 0,
            'node_types': defaultdict(int),
            'comm_types': defaultdict(int),
            'total_ops': 0,
            'total_tensor_size': 0,
            'total_comm_size': 0,
            'max_depth': 0,
        }
        
    def read_et_file(self):
        """Read ET file"""
        print(f"Reading ET file: {self.input_file}")
        
        try:
            with open(self.input_file, "rb") as f:
                # Read global metadata
                self.metadata = GlobalMetadata()
                if not decode_message(f, self.metadata):
                    raise RuntimeError("Failed to read global metadata")
                
                print(f"   Metadata version: {self.metadata.version}")
                
                # Read all nodes
                node = ChakraNode()
                node_count = 0
                while decode_message(f, node):
                    self.nodes.append(self._copy_node(node))
                    node_count += 1
                    if node_count % 100 == 0:
                        print(f"   Reading nodes... {node_count}", end='\r')
                
                print(f"   Total nodes read: {node_count}      ")
                
        except FileNotFoundError:
            print(f"Error: File not found: {self.input_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    def _copy_node(self, node: ChakraNode) -> Dict:
        """Copy node data to dictionary"""
        node_dict = {
            'id': node.id,
            'name': node.name,
            'type': node.type,
            'ctrl_deps': list(node.ctrl_deps),
            'data_deps': list(node.data_deps),
            'duration_micros': node.duration_micros,
            'attrs': {},
        }
        
        # Parse attributes
        for attr in node.attr:
            if attr.HasField('bool_val'):
                node_dict['attrs'][attr.name] = attr.bool_val
            elif attr.HasField('int64_val'):
                node_dict['attrs'][attr.name] = attr.int64_val
            elif attr.HasField('uint64_val'):
                node_dict['attrs'][attr.name] = attr.uint64_val
            elif attr.HasField('string_val'):
                node_dict['attrs'][attr.name] = attr.string_val
            elif attr.HasField('double_val'):
                node_dict['attrs'][attr.name] = attr.double_val
        
        return node_dict
    
    def analyze(self):
        """Analyze ET file content"""
        print(f"\nAnalyzing execution trace...")
        
        for node in self.nodes:
            self.stats['total_nodes'] += 1
            
            # Count node types
            node_type = node['type']
            node_type_name = self.NODE_TYPE_NAMES.get(node_type, f"UNKNOWN_{node_type}")
            self.stats['node_types'][node_type_name] += 1
            
            # Analyze attributes
            attrs = node['attrs']
            
            # Computation nodes
            if node_type == COMP_NODE:
                if 'num_ops' in attrs:
                    self.stats['total_ops'] += attrs['num_ops']
                if 'tensor_size' in attrs:
                    self.stats['total_tensor_size'] += attrs['tensor_size']
            
            # Communication nodes
            if node_type == COMM_COLL_NODE:
                if 'comm_type' in attrs:
                    comm_type = attrs['comm_type']
                    comm_name = self.COMM_TYPE_NAMES.get(comm_type, f"UNKNOWN_{comm_type}")
                    self.stats['comm_types'][comm_name] += 1
                
                if 'comm_size' in attrs:
                    self.stats['total_comm_size'] += attrs['comm_size']
            
            # Calculate graph depth (max dependencies)
            depth = len(node['ctrl_deps']) + len(node['data_deps'])
            self.stats['max_depth'] = max(self.stats['max_depth'], depth)
        
        print(f"   Analysis complete!")
    
    def print_summary(self, output_file=None):
        """Print analysis summary"""
        output = []
        
        output.append("=" * 80)
        output.append("CHAKRA ET FILE ANALYSIS REPORT")
        output.append("=" * 80)
        output.append(f"Input File: {self.input_file}")
        output.append(f"Chakra Version: {self.metadata.version if self.metadata else 'Unknown'}")
        output.append("")
        
        # Basic statistics
        output.append("BASIC STATISTICS")
        output.append("-" * 80)
        output.append(f"Total Nodes:           {self.stats['total_nodes']:,}")
        output.append(f"Max Dependency Depth:  {self.stats['max_depth']}")
        output.append("")
        
        # Node type breakdown
        output.append("NODE TYPE BREAKDOWN")
        output.append("-" * 80)
        output.append(f"{'Node Type':<20} {'Count':>12} {'Percentage':>12}")
        output.append("-" * 80)
        for node_type, count in sorted(self.stats['node_types'].items(), 
                                       key=lambda x: x[1], reverse=True):
            percentage = 100.0 * count / self.stats['total_nodes'] if self.stats['total_nodes'] > 0 else 0
            output.append(f"{node_type:<20} {count:>12,} {percentage:>11.2f}%")
        output.append("")
        
        # Communication operations
        if self.stats['comm_types']:
            output.append("COMMUNICATION OPERATIONS")
            output.append("-" * 80)
            output.append(f"{'Collective Type':<20} {'Count':>12}")
            output.append("-" * 80)
            for comm_type, count in sorted(self.stats['comm_types'].items(), 
                                           key=lambda x: x[1], reverse=True):
                output.append(f"{comm_type:<20} {count:>12,}")
            output.append("")
            output.append(f"Total Communication Size: {self._format_bytes(self.stats['total_comm_size'])}")
            output.append("")
        
        # Computation operations
        if self.stats['total_ops'] > 0:
            output.append("COMPUTATION OPERATIONS")
            output.append("-" * 80)
            output.append(f"Total Operations:      {self._format_ops(self.stats['total_ops'])}")
            output.append(f"Total Tensor Size:     {self._format_bytes(self.stats['total_tensor_size'])}")
            output.append("")
        
        # Sample nodes
        output.append("SAMPLE NODES (First 5)")
        output.append("-" * 80)
        for i, node in enumerate(self.nodes[:5]):
            output.append(f"Node {i+1}:")
            output.append(f"  ID:              {node['id']}")
            output.append(f"  Name:            {node['name']}")
            output.append(f"  Type:            {self.NODE_TYPE_NAMES.get(node['type'], 'UNKNOWN')}")
            output.append(f"  Control Deps:    {len(node['ctrl_deps'])}")
            output.append(f"  Data Deps:       {len(node['data_deps'])}")
            if node['attrs']:
                output.append(f"  Attributes:      {list(node['attrs'].keys())}")
            output.append("")
        
        output.append("=" * 80)
        
        # Output to file or console
        report = "\n".join(output)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        else:
            print(report)
    
    def print_detailed_stats(self):
        """Print detailed statistics"""
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS")
        print("=" * 80)
        
        # Analyze node names
        name_prefix_counts = defaultdict(int)
        for node in self.nodes:
            name = node['name']
            if '.' in name:
                prefix = name.split('.')[0]
                name_prefix_counts[prefix] += 1
        
        if name_prefix_counts:
            print("\nNODE NAME PREFIXES")
            print("-" * 80)
            for prefix, count in sorted(name_prefix_counts.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                print(f"{prefix:<30} {count:>10,}")
        
        # Communication size distribution
        if self.stats['total_comm_size'] > 0:
            comm_sizes = []
            for node in self.nodes:
                if node['type'] == COMM_COLL_NODE and 'comm_size' in node['attrs']:
                    comm_sizes.append(node['attrs']['comm_size'])
            
            if comm_sizes:
                print("\nCOMMUNICATION SIZE DISTRIBUTION")
                print("-" * 80)
                print(f"Min:     {self._format_bytes(min(comm_sizes))}")
                print(f"Max:     {self._format_bytes(max(comm_sizes))}")
                print(f"Average: {self._format_bytes(sum(comm_sizes) // len(comm_sizes))}")
        
        print("\n" + "=" * 80)
    
    def export_graph_info(self, output_file: str):
        """Export graph structure information"""
        print(f"\nExporting graph information to: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Chakra ET Graph Structure\n")
            f.write(f"# Total Nodes: {len(self.nodes)}\n\n")
            
            for node in self.nodes:
                f.write(f"Node {node['id']}: {node['name']}\n")
                f.write(f"  Type: {self.NODE_TYPE_NAMES.get(node['type'], 'UNKNOWN')}\n")
                
                if node['ctrl_deps']:
                    f.write(f"  Control Deps: {node['ctrl_deps']}\n")
                if node['data_deps']:
                    f.write(f"  Data Deps: {node['data_deps']}\n")
                
                # Print key attributes
                if node['type'] == COMP_NODE and 'num_ops' in node['attrs']:
                    ops = node['attrs']['num_ops']
                    f.write(f"  Operations: {self._format_ops(ops)}\n")
                
                if node['type'] == COMM_COLL_NODE:
                    if 'comm_type' in node['attrs']:
                        comm_type = self.COMM_TYPE_NAMES.get(node['attrs']['comm_type'], 'UNKNOWN')
                        f.write(f"  Comm Type: {comm_type}\n")
                    if 'comm_size' in node['attrs']:
                        size = node['attrs']['comm_size']
                        f.write(f"  Comm Size: {self._format_bytes(size)}\n")
                
                f.write("\n")
        
        print(f"   Graph info exported!")
    
    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"
    
    @staticmethod
    def _format_ops(num_ops: int) -> str:
        """Format operations"""
        if num_ops < 1e3:
            return f"{num_ops:.0f} Ops"
        elif num_ops < 1e6:
            return f"{num_ops/1e3:.2f} K Ops"
        elif num_ops < 1e9:
            return f"{num_ops/1e6:.2f} M Ops"
        elif num_ops < 1e12:
            return f"{num_ops/1e9:.2f} G Ops (GFLOPs)"
        else:
            return f"{num_ops/1e12:.2f} T Ops (TFLOPs)"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Chakra Execution Trace (.et) files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_et_file.py --input attention_tp8.0.et
  
  # Save report to file
  python analyze_et_file.py --input attention_tp8.0.et --output report.txt
  
  # Show detailed statistics
  python analyze_et_file.py --input attention_tp8.0.et --stats
  
  # Export graph structure
  python analyze_et_file.py --input attention_tp8.0.et --graph graph_info.txt
  
  # All options
  python analyze_et_file.py --input attention_tp8.0.et --output report.txt --stats --graph graph.txt
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input Chakra ET file (.et)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output report file (default: print to console)'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show detailed statistics'
    )
    
    parser.add_argument(
        '--graph', '-g',
        type=str,
        help='Export graph structure to file'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ETAnalyzer(args.input)
    
    # Read and analyze
    analyzer.read_et_file()
    analyzer.analyze()
    
    # Print summary
    analyzer.print_summary(args.output)
    
    # Detailed stats
    if args.stats:
        analyzer.print_detailed_stats()
    
    # Export graph
    if args.graph:
        analyzer.export_graph_info(args.graph)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
