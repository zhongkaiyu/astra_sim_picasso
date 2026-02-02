# Scripts 工具索引

本目录包含用于分析 Chakra ET 文件的各种工具和脚本。

## ? 工具列表

### 1. ET 文件分析器（核心工具）

**文件**: `analyze_et_file.py`  
**用途**: 分析单个 Chakra Execution Trace (.et) 文件  
**语言**: Python 3  

**功能**:
- ? 读取和解析 Chakra ET 文件（protobuf 格式）
- ? 统计节点类型分布（COMP, COMM, MEM 等）
- ? 分析通信操作（AllReduce, AllGather 等）
- ? 统计计算量（GFLOPs）和数据量（MB/GB）
- ? 生成详细的分析报告
- ? 导出图结构信息

**快速开始**:
```bash
python analyze_et_file.py --input ../output/megatron_tp8/workloads/attention_tp8.0.et
```

---

### 2. 批量分析脚本

**文件**: `batch_analyze_et.sh`  
**用途**: 批量分析一组 ET 文件并生成对比报告  
**语言**: Bash  

**功能**:
- ? 自动分析多个 ET 文件
- ? 生成汇总报告和对比数据
- ? 导出 CSV 格式的统计数据
- ? 适合分析所有 NPU 的 workload

**快速开始**:
```bash
bash batch_analyze_et.sh -i ../output/megatron_tp8/workloads -p attention_tp8 -n 8
```

---

### 3. 交互式演示脚本

**文件**: `demo_analyze_et.sh`  
**用途**: 交互式演示如何使用 ET 分析工具  
**语言**: Bash  

**功能**:
- ? 分步演示所有功能
- ? 自动查找示例文件
- ? 展示实际输出结果
- ? 提供使用提示

**快速开始**:
```bash
bash demo_analyze_et.sh
```

---

## ? 文档

### README_ANALYZE_ET.md
完整的使用指南，包含：
- 详细的功能说明
- 命令行参数解释
- 使用示例和模板
- 常见问题解答
- 高级用法和技巧

### TOOLS_INDEX.md（本文件）
工具索引和快速参考

---

## ? 快速使用场景

### 场景 1: 检查 workload 是否生成正确
```bash
# 分析生成的 workload
python analyze_et_file.py --input ../symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8.0.et

# 查看节点数量和类型分布
```

### 场景 2: 对比不同 NPU 的 workload
```bash
# 批量分析所有 NPU
bash batch_analyze_et.sh -i ../output/megatron_tp8/workloads -p attention_tp8 -n 8

# 查看 comparison.csv
cat analysis_results/comparison.csv
```

### 场景 3: 调试通信问题
```bash
# 详细分析通信操作
python analyze_et_file.py --input attention_tp8.0.et --stats | grep -A 10 "COMMUNICATION"
```

### 场景 4: 提取图结构
```bash
# 导出图结构用于可视化
python analyze_et_file.py --input attention_tp8.0.et --graph graph.txt
```

### 场景 5: 生成报告文档
```bash
# 生成完整的分析报告
python analyze_et_file.py --input attention_tp8.0.et --output report.txt --stats
```

---

## ? 环境要求

### Python 依赖
```bash
# 需要安装 Chakra
cd ../extern/graph_frontend/chakra
pip install -e .
```

### 系统要求
- Python 3.7+
- protobuf 3.x+
- Bash (for batch scripts)

---

## ? 输出文件说明

### analyze_et_file.py 的输出

| 选项 | 输出文件 | 内容 |
|------|---------|------|
| 默认 | 控制台 | 基本分析报告 |
| `--output` | 文本文件 | 完整分析报告 |
| `--stats` | 控制台 | 详细统计信息 |
| `--graph` | 文本文件 | 图结构信息 |

### batch_analyze_et.sh 的输出

| 文件 | 内容 |
|------|------|
| `summary.txt` | 所有 NPU 的汇总信息 |
| `comparison.csv` | 对比数据（CSV 格式） |
| `report_npuX.txt` | 每个 NPU 的详细报告 |

---

## ? 与 ASTRA-sim 工作流集成

```bash
# 完整的分析流程

# 1. 生成 workload
cd ../symbolic_tensor_graph_picasso
python main.py --output_dir generated_attn_8npu/ --output_name attention_tp8 \
    --model_type moe_attention --tp 8 --num_stacks 1 --dmodel 4096 \
    --head 64 --kvhead 4 --batch 1 --seq 1024

# 2. 分析 workload（运行模拟前）
cd ../scripts
python analyze_et_file.py --input ../symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8.0.et \
    --output pre_simulation_analysis.txt

# 3. 运行 ASTRA-sim 模拟
cd ../examples/run_scripts/analytical/congestion_aware
bash Megatron_TP8_attention.sh

# 4. 分析模拟后的 workload（如果有的话）
cd ../../../../scripts
python analyze_et_file.py --input ../output/megatron_tp8/workloads/attention_tp8.0.et \
    --output post_simulation_analysis.txt

# 5. 对比分析
diff pre_simulation_analysis.txt post_simulation_analysis.txt
```

---

## ? 使用技巧

### 1. 快速查看节点数量
```bash
python analyze_et_file.py -i file.et | grep "Total Nodes"
```

### 2. 只查看通信操作
```bash
python analyze_et_file.py -i file.et | grep -A 10 "COMMUNICATION"
```

### 3. 导出所有信息
```bash
python analyze_et_file.py -i file.et -o report.txt -s -g graph.txt
```

### 4. 比较两个 workload
```bash
python analyze_et_file.py -i workload1.et > w1.txt
python analyze_et_file.py -i workload2.et > w2.txt
diff w1.txt w2.txt
```

---

## ? 故障排除

### 问题 1: 找不到 Chakra 模块
```bash
# 解决方法：安装 Chakra
cd ../extern/graph_frontend/chakra
pip install -e .
```

### 问题 2: protobuf 版本不兼容
```bash
# 解决方法：升级 protobuf
pip install --upgrade protobuf
```

### 问题 3: 权限错误
```bash
# 解决方法：添加执行权限
chmod +x *.sh *.py
```

---

## ? 获取帮助

```bash
# 查看工具帮助
python analyze_et_file.py --help
bash batch_analyze_et.sh --help

# 运行演示
bash demo_analyze_et.sh

# 阅读完整文档
cat README_ANALYZE_ET.md
```

---

## ? 相关资源

- **Chakra 文档**: `../extern/graph_frontend/chakra/README.md`
- **ASTRA-sim 文档**: `../README.md`
- **Workload 生成**: `../symbolic_tensor_graph_picasso/README.md`
- **快速参考**: `../output/ET_ANALYSIS_QUICKREF.md`

---

**最后更新**: 2025-12-22  
**作者**: ASTRA-sim Team



