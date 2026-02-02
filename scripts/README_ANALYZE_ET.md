# ET 文件分析工具使用指南

## 工具简介

`analyze_et_file.py` 是一个用于分析 Chakra Execution Trace (.et) 文件的 Python 脚本。

## 功能特性

- ? 读取和解析 Chakra ET 文件（protobuf 格式）
- ? 统计节点类型分布（计算、通信、内存等）
- ? 分析通信操作（AllReduce, AllGather 等）
- ? 统计计算量和数据量
- ? 生成详细的分析报告
- ? 导出图结构信息

## 快速开始

### 基本使用
```bash
cd scripts
python analyze_et_file.py --input ../output/megatron_tp8/workloads/attention_tp8.0.et
```

### 保存报告到文件
```bash
python analyze_et_file.py \
    --input ../output/megatron_tp8/workloads/attention_tp8.0.et \
    --output analysis_report.txt
```

### 显示详细统计
```bash
python analyze_et_file.py \
    --input ../output/megatron_tp8/workloads/attention_tp8.0.et \
    --stats
```

### 导出图结构
```bash
python analyze_et_file.py \
    --input ../output/megatron_tp8/workloads/attention_tp8.0.et \
    --graph graph_structure.txt
```

### 完整分析（所有选项）
```bash
python analyze_et_file.py \
    --input ../output/megatron_tp8/workloads/attention_tp8.0.et \
    --output report.txt \
    --stats \
    --graph graph.txt
```

## 命令行参数

| 参数 | 简写 | 必需 | 说明 |
|------|------|------|------|
| `--input` | `-i` | ? | 输入的 .et 文件路径 |
| `--output` | `-o` | ? | 输出报告文件路径（默认：控制台） |
| `--stats` | `-s` | ? | 显示详细统计信息 |
| `--graph` | `-g` | ? | 导出图结构到文件 |

## 输出示例

### 基本报告
```
================================================================================
CHAKRA ET FILE ANALYSIS REPORT
================================================================================
Input File: attention_tp8.0.et
Chakra Version: 0.0.4

? BASIC STATISTICS
--------------------------------------------------------------------------------
Total Nodes:           442
Max Dependency Depth:  5

? NODE TYPE BREAKDOWN
--------------------------------------------------------------------------------
Node Type            Count    Percentage
--------------------------------------------------------------------------------
COMP                   250        56.56%
COMM_COLL              100        22.62%
METADATA                92        20.82%

? COMMUNICATION OPERATIONS
--------------------------------------------------------------------------------
Collective Type      Count
--------------------------------------------------------------------------------
AllReduce               50
AllGather               30
ReduceScatter           20

Total Communication Size: 512.00 MB

? COMPUTATION OPERATIONS
--------------------------------------------------------------------------------
Total Operations:      125.50 GFLOPs
Total Tensor Size:     2.50 GB
```

## 批量分析脚本

### 分析所有 TP8 workload
```bash
#!/bin/bash
for i in {0..7}; do
    echo "Analyzing NPU $i..."
    python analyze_et_file.py \
        --input ../output/megatron_tp8/workloads/attention_tp8.$i.et \
        --output report_npu$i.txt
done
```

### 比较不同 NPU 的 workload
```bash
#!/bin/bash
for i in {0..7}; do
    python analyze_et_file.py \
        --input ../output/megatron_tp8/workloads/attention_tp8.$i.et \
        | grep "Total Nodes" >> comparison.txt
done
```

## 常见问题

### Q1: 找不到 Chakra 模块
**问题**: `ImportError: No module named 'chakra'`

**解决**:
```bash
# 确保 Chakra 已安装
cd extern/graph_frontend/chakra
pip install -e .
```

### Q2: 文件路径错误
**问题**: `FileNotFoundError`

**解决**: 使用绝对路径或相对于 scripts 目录的正确路径
```bash
# 相对路径（从 scripts 目录）
python analyze_et_file.py --input ../output/megatron_tp8/workloads/attention_tp8.0.et

# 绝对路径
python analyze_et_file.py --input /full/path/to/attention_tp8.0.et
```

### Q3: Protobuf 版本不兼容
**问题**: `protobuf version mismatch`

**解决**:
```bash
pip install --upgrade protobuf
```

## 高级用法

### 1. 自动生成对比报告
```bash
# 创建 compare_workloads.sh
#!/bin/bash
OUTPUT_DIR="analysis_results"
mkdir -p $OUTPUT_DIR

for i in {0..7}; do
    echo "=== NPU $i ===" >> $OUTPUT_DIR/summary.txt
    python analyze_et_file.py \
        --input ../output/megatron_tp8/workloads/attention_tp8.$i.et \
        | grep -E "(Total Nodes|COMP|COMM_COLL)" >> $OUTPUT_DIR/summary.txt
    echo "" >> $OUTPUT_DIR/summary.txt
done
```

### 2. 提取通信模式
```bash
# 只查看通信操作
python analyze_et_file.py \
    --input attention_tp8.0.et \
    | grep -A 10 "COMMUNICATION OPERATIONS"
```

### 3. 统计计算量
```bash
# 只查看计算操作
python analyze_et_file.py \
    --input attention_tp8.0.et \
    | grep -A 5 "COMPUTATION OPERATIONS"
```

## 与其他工具集成

### 与 ASTRA-sim 日志对比
```bash
# 分析 workload
python analyze_et_file.py --input attention_tp8.0.et --output workload_analysis.txt

# 对比模拟结果
grep "Total execution" ../output/megatron_tp8/logs/simulation_log_megatron_tp8.txt
```

### 生成可视化数据
```bash
# 导出图结构后，可以用其他工具可视化
python analyze_et_file.py --input attention_tp8.0.et --graph graph.txt

# 使用 Graphviz 或其他工具处理 graph.txt
```

## 性能提示

- 对于大文件（>100MB），读取可能需要几秒钟
- 使用 `--output` 保存到文件比打印到控制台更快
- `--stats` 选项会增加额外的分析时间

## 相关文件

- 脚本位置: `scripts/analyze_et_file.py`
- 示例 workload: `output/megatron_tp8/workloads/`
- Chakra 文档: `extern/graph_frontend/chakra/README.md`

## 许可证

MIT License



