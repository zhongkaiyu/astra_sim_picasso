#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTRA-sim Log Analyzer
======================
Analyzes ASTRA-sim simulation logs and extracts key performance metrics.

Usage:
    python analyze_log.py <log_file>
    python analyze_log.py <log_file1> <log_file2> --compare

Author: Generated for ASTRA-sim analysis
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional


class AstraSimLogAnalyzer:
    """Analyzer for ASTRA-sim simulation logs."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.metrics = {}
        self.all_sys_metrics = {}
        
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
    
    def parse_log(self) -> Dict:
        """Parse the log file and extract metrics."""
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        # Extract sys[0] metrics (representative NPU)
        try:
            self.metrics = {
                'wall_time': self._extract_value(content, r'sys\[0\].*Wall time: (\d+)'),
                'gpu_time': self._extract_value(content, r'sys\[0\].*GPU time: (\d+)'),
                'comm_time': self._extract_value(content, r'sys\[0\].*Comm time: (\d+)'),
                'overlap': self._extract_value(content, r'sys\[0\].*Total compute-communication overlap: (\d+)', default=0),
                'compute_bound_pct': self._extract_float(content, r'sys\[0\].*Compute bound percentage: ([\d.]+)'),
                'compute_util': self._extract_float(content, r'sys\[0\].*Average compute utilization: ([\d.]+)'),
                'memory_util': self._extract_float(content, r'sys\[0\].*Average memory utilization: ([\d.]+)'),
                'operation_intensity': self._extract_float(content, r'sys\[0\].*Average operation intensity: ([\d.]+)', default=0),
            }
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return {}
        
        # Calculate derived metrics
        self._calculate_derived_metrics()
        
        # Extract all NPU metrics for consistency check
        self._extract_all_npu_metrics(content)
        
        return self.metrics
    
    def _extract_value(self, content: str, pattern: str, default: int = 0) -> int:
        """Extract integer value from log using regex pattern."""
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
        return default
    
    def _extract_float(self, content: str, pattern: str, default: float = 0.0) -> float:
        """Extract float value from log using regex pattern."""
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        return default
    
    def _calculate_derived_metrics(self):
        """Calculate percentage and ratio metrics."""
        wall_time = self.metrics['wall_time']
        gpu_time = self.metrics['gpu_time']
        comm_time = self.metrics['comm_time']
        
        if wall_time > 0:
            self.metrics['gpu_pct'] = (gpu_time / wall_time) * 100
            self.metrics['comm_pct'] = (comm_time / wall_time) * 100
            self.metrics['overlap_pct'] = (self.metrics['overlap'] / wall_time) * 100
        else:
            self.metrics['gpu_pct'] = 0
            self.metrics['comm_pct'] = 0
            self.metrics['overlap_pct'] = 0
    
    def _extract_all_npu_metrics(self, content: str):
        """Extract metrics for all NPUs to check consistency."""
        npu_pattern = r'sys\[(\d+)\].*Wall time: (\d+)'
        matches = re.findall(npu_pattern, content)
        
        for npu_id, wall_time in matches:
            self.all_sys_metrics[int(npu_id)] = int(wall_time)
    
    def identify_bottleneck(self) -> str:
        """Identify the main performance bottleneck."""
        comm_pct = self.metrics['comm_pct']
        gpu_pct = self.metrics['gpu_pct']
        
        if comm_pct > 90:
            return "SEVERE_COMM_BOUND"
        elif comm_pct > 70:
            return "COMM_BOUND"
        elif gpu_pct > 60:
            return "COMPUTE_BOUND"
        else:
            return "BALANCED"
    
    def get_bottleneck_description(self) -> str:
        """Get human-readable bottleneck description."""
        bottleneck = self.identify_bottleneck()
        descriptions = {
            "SEVERE_COMM_BOUND": "[WARNING] Severely communication-bound (>90%)",
            "COMM_BOUND": "[WARNING] Communication-bound (70-90%)",
            "COMPUTE_BOUND": "[OK] Compute-bound (GPU >60%)",
            "BALANCED": "[BALANCED] Compute and communication both significant"
        }
        return descriptions.get(bottleneck, "Unknown")
    
    def print_report(self, output_file=None):
        """Print formatted analysis report."""
        if not self.metrics:
            print("No metrics available. Please parse log first.")
            return
        
        # Determine output destination
        if output_file:
            import sys
            original_stdout = sys.stdout
            sys.stdout = open(output_file, 'w')
        
        print("=" * 70)
        print(f"ASTRA-sim Log Analysis: {self.log_file.name}")
        print("=" * 70)
        print()
        
        # Basic metrics
        print("[Performance Metrics]")
        print("-" * 70)
        print(f"  Wall Time:           {self.metrics['wall_time']:>15,} cycles (100%)")
        print(f"  GPU Time (Compute):  {self.metrics['gpu_time']:>15,} cycles ({self.metrics['gpu_pct']:>5.2f}%)")
        print(f"  Comm Time:           {self.metrics['comm_time']:>15,} cycles ({self.metrics['comm_pct']:>5.2f}%)")
        print(f"  Compute-Comm Overlap:{self.metrics['overlap']:>15,} cycles ({self.metrics['overlap_pct']:>5.2f}%)")
        print()
        
        # Utilization metrics
        print("[Utilization Metrics]")
        print("-" * 70)
        print(f"  Compute Bound:       {self.metrics['compute_bound_pct']:>15.2f}%")
        print(f"  Compute Utilization: {self.metrics['compute_util']:>15.2f}%")
        print(f"  Memory Utilization:  {self.metrics['memory_util']:>15.2f}%")
        print(f"  Operation Intensity: {self.metrics['operation_intensity']:>15,.2f}")
        print()
        
        # Bottleneck analysis
        print("[Bottleneck Analysis]")
        print("-" * 70)
        print(f"  Status: {self.get_bottleneck_description()}")
        print()
        
        # Time breakdown visualization
        print("[Time Breakdown]")
        print("-" * 70)
        self._print_bar_chart("Compute", self.metrics['gpu_pct'])
        self._print_bar_chart("Communication", self.metrics['comm_pct'])
        print()
        
        # NPU consistency check
        if len(self.all_sys_metrics) > 1:
            wall_times = list(self.all_sys_metrics.values())
            min_time = min(wall_times)
            max_time = max(wall_times)
            avg_time = sum(wall_times) / len(wall_times)
            variance = ((max_time - min_time) / avg_time) * 100
            
            print("[Multi-NPU Consistency Check]")
            print("-" * 70)
            print(f"  Total NPUs:          {len(self.all_sys_metrics)}")
            print(f"  Min Wall Time:       {min_time:,} cycles")
            print(f"  Max Wall Time:       {max_time:,} cycles")
            print(f"  Avg Wall Time:       {avg_time:,.0f} cycles")
            print(f"  Variance:            {variance:.2f}%")
            
            if variance < 1:
                print(f"  Status:              [OK] Excellent load balance (<1% variance)")
            elif variance < 5:
                print(f"  Status:              [OK] Good load balance (<5% variance)")
            else:
                print(f"  Status:              [WARNING] Load imbalance detected (>{variance:.1f}% variance)")
            print()
        
        # Restore stdout if redirected
        if output_file:
            import sys
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"[OK] Analysis saved to: {output_file}")
    
    def _print_bar_chart(self, label: str, percentage: float, width: int = 50):
        """Print a simple bar chart."""
        filled = int((percentage / 100) * width)
        bar = "#" * filled + "-" * (width - filled)
        print(f"  {label:15s} {percentage:5.1f}% {bar}")
    
    def get_summary_dict(self) -> Dict:
        """Return metrics as a dictionary for comparison."""
        return {
            'file': self.log_file.name,
            'wall_time': self.metrics['wall_time'],
            'gpu_time': self.metrics['gpu_time'],
            'comm_time': self.metrics['comm_time'],
            'gpu_pct': self.metrics['gpu_pct'],
            'comm_pct': self.metrics['comm_pct'],
            'compute_util': self.metrics['compute_util'],
            'bottleneck': self.identify_bottleneck()
        }


def compare_logs(log_files: List[str], output_file=None):
    """Compare multiple log files."""
    analyzers = []
    summaries = []
    
    # Redirect output if needed
    if output_file:
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w')
    
    print("=" * 70)
    print("ASTRA-sim Multi-Configuration Comparison")
    print("=" * 70)
    print()
    
    # Analyze each log
    for log_file in log_files:
        try:
            analyzer = AstraSimLogAnalyzer(log_file)
            analyzer.parse_log()
            analyzers.append(analyzer)
            summaries.append(analyzer.get_summary_dict())
            print(f"[OK] Loaded: {log_file}")
        except Exception as e:
            print(f"[ERROR] Error loading {log_file}: {e}")
    
    if len(summaries) < 2:
        print("\nNeed at least 2 valid log files for comparison.")
        return
    
    print()
    print("=" * 70)
    print("Comparison Table")
    print("=" * 70)
    
    # Print header
    print(f"{'Metric':<25}", end='')
    for i, summary in enumerate(summaries):
        print(f" {'Config ' + str(i+1):>15}", end='')
    print()
    print("-" * 70)
    
    # Print metrics
    metrics_to_compare = [
        ('Wall Time (cycles)', 'wall_time', ','),
        ('GPU Time (cycles)', 'gpu_time', ','),
        ('Comm Time (cycles)', 'comm_time', ','),
        ('GPU %', 'gpu_pct', '.2f'),
        ('Comm %', 'comm_pct', '.2f'),
        ('Compute Util %', 'compute_util', '.2f'),
        ('Bottleneck', 'bottleneck', 's'),
    ]
    
    for label, key, fmt in metrics_to_compare:
        print(f"{label:<25}", end='')
        for summary in summaries:
            value = summary[key]
            if fmt == ',':
                print(f" {value:>15,}", end='')
            elif fmt == '.2f':
                print(f" {value:>15.2f}", end='')
            else:
                print(f" {value:>15}", end='')
        print()
    
    print()
    print("=" * 70)
    print("Speedup Analysis (vs Config 1)")
    print("=" * 70)
    
    baseline_time = summaries[0]['wall_time']
    for i, summary in enumerate(summaries):
        if i == 0:
            print(f"  Config 1: {summary['file']}")
            print(f"    Baseline: {baseline_time:,} cycles")
        else:
            speedup = baseline_time / summary['wall_time']
            time_saved = (1 - summary['wall_time'] / baseline_time) * 100
            print(f"  Config {i+1}: {summary['file']}")
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Time saved: {time_saved:.1f}%")
    print()
    
    # Restore stdout if redirected
    if output_file:
        import sys
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"[OK] Comparison saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ASTRA-sim simulation logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single log (auto-saves to analysis/)
  python analyze_log.py output/megatron_tp8/logs/simulation_log_megatron_tp8.txt
  
  # Compare two logs (auto-saves to output/comparison/)
  python analyze_log.py output/mesh2d_tp24/logs/simulation_log.txt \\
                        output/megatron_tp8/logs/simulation_log_megatron_tp8.txt \\
                        --compare
  
  # Custom output path
  python analyze_log.py log.txt --output my_analysis.txt
        """
    )
    
    parser.add_argument('log_files', nargs='+', help='Path to log file(s)')
    parser.add_argument('--compare', '-c', action='store_true', 
                       help='Compare multiple log files')
    parser.add_argument('--output', '-o', help='Save report to file (default: auto-generate path)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output, only print to console')
    
    args = parser.parse_args()
    
    # Determine output file path
    output_file = None
    if not args.no_save:
        if args.output:
            output_file = args.output
        else:
            # Auto-generate output path
            if args.compare and len(args.log_files) > 1:
                output_file = "output/comparison/analysis_comparison.txt"
            elif len(args.log_files) == 1:
                log_path = Path(args.log_files[0])
                # Try to save in same directory structure
                if 'output' in str(log_path) and 'logs' in str(log_path):
                    # Replace 'logs' with 'analysis'
                    output_file = str(log_path).replace('/logs/', '/analysis/').replace('.txt', '_analysis.txt')
                else:
                    # Save in current directory
                    output_file = f"{log_path.stem}_analysis.txt"
    
    if args.compare and len(args.log_files) > 1:
        compare_logs(args.log_files, output_file)
    elif len(args.log_files) == 1:
        try:
            analyzer = AstraSimLogAnalyzer(args.log_files[0])
            analyzer.parse_log()
            analyzer.print_report(output_file)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print("Error: Provide either one log file or multiple with --compare")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

