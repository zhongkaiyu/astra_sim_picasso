# analyze_log.py 使用指南

## 📋 功能概述

`analyze_log.py` 是一个专门用于分析 ASTRA-sim 仿真日志的工具，它能自动提取关键性能指标并生成可读的分析报告。

## 🎯 抓取的内容

### 1. 核心性能指标（从 sys[0] 提取）

脚本从日志文件中提取以下关键指标：

```python
# 时间指标
- Wall Time (墙钟时间)        # 总执行时间
- GPU Time (计算时间)         # GPU计算耗时
- Comm Time (通信时间)        # 网络通信耗时
- Overlap (计算-通信重叠)     # 计算和通信重叠部分

# 利用率指标
- Compute Bound Percentage    # 计算受限百分比
- Compute Utilization         # 计算资源利用率
- Memory Utilization          # 内存利用率
- Operation Intensity         # 操作强度 (FLOPs/byte)
```

### 2. 实际提取位置（日志示例）

从日志文件的统计信息部分提取：

```
[statistics] [info] sys[0], Wall time: 4932010           ← 提取
[statistics] [info] sys[0], GPU time: 3154090            ← 提取
[statistics] [info] sys[0], Comm time: 4846452           ← 提取
[statistics] [info] sys[0], Total compute-communication overlap: 3068532  ← 提取
[statistics] [info] sys[0], Compute bound percentage: 86.799%    ← 提取
[statistics] [info] sys[0], Average compute utilization: 89.091% ← 提取
[statistics] [info] sys[0], Average memory utilization: 30.748%  ← 提取
[statistics] [info] sys[0], Average operation intensity: 2491.463 ← 提取
```

### 3. 衍生指标（自动计算）

脚本自动计算以下百分比：

```python
- GPU %     = (GPU Time / Wall Time) × 100
- Comm %    = (Comm Time / Wall Time) × 100
- Overlap % = (Overlap / Wall Time) × 100
```

### 4. 多NPU一致性检查

提取所有 NPU 的 Wall Time 并比较：

```python
- Min/Max/Avg Wall Time  # 检查负载均衡
- Variance %             # 变异系数
```

## ✅ 抓取是否合理？

### 优点

1. **✓ 关键指标完整**
   - 覆盖了计算、通信、重叠三大核心维度
   - 包含利用率和性能强度指标

2. **✓ 只提取 sys[0] 作为代表**
   - 合理：在负载均衡的情况下，所有 NPU 指标相同
   - 效率：避免重复提取相同数据
   - 验证：通过多NPU一致性检查确认这个假设

3. **✓ 自动瓶颈分析**
   - 根据通信/计算占比自动识别瓶颈
   - 给出明确的优化建议方向

4. **✓ 可视化输出**
   - 条形图直观显示时间分布
   - 百分比便于快速理解

### 可能的改进点

1. **⚠ 缺少详细的通信模式分析**
   - 建议添加：AllReduce、AllGather 等不同集合通信的时间分解
   - 建议添加：网络带宽利用率

2. **⚠ 没有提取层级（Layer）信息**
   - 日志中有详细的 transformer layer 信息
   - 可以添加逐层分析功能

3. **⚠ 没有提取算子级（Node）统计**
   - 日志中有每个算子的执行时间
   - 可以添加热点算子分析

## 📖 用法

### 1. 基本用法：分析单个日志

```bash
# 方式1：自动保存到 analysis/ 目录
python analyze_log.py output/tp4_h100/logs/simulation_log_tp4_h100.txt

# 方式2：只显示不保存
python analyze_log.py output/tp4_h100/logs/simulation_log_tp4_h100.txt --no-save

# 方式3：指定输出路径
python analyze_log.py output/tp4_h100/logs/simulation_log_tp4_h100.txt --output my_analysis.txt
```

**输出示例：**

```
======================================================================
ASTRA-sim Log Analysis: simulation_log_tp4_h100.txt
======================================================================

[Performance Metrics]
----------------------------------------------------------------------
  Wall Time:                 4,932,010 cycles (100%)
  GPU Time (Compute):        3,154,090 cycles (63.95%)
  Comm Time:                 4,846,452 cycles (98.27%)
  Compute-Comm Overlap:      3,068,532 cycles (62.22%)

[Utilization Metrics]
----------------------------------------------------------------------
  Compute Bound:                 86.80%
  Compute Utilization:           89.09%
  Memory Utilization:            30.75%
  Operation Intensity:        2,491.46

[Bottleneck Analysis]
----------------------------------------------------------------------
  Status: [WARNING] Severely communication-bound (>90%)

[Time Breakdown]
----------------------------------------------------------------------
  Compute          64.0% ###############################-------------------
  Communication    98.3% #################################################-

[Multi-NPU Consistency Check]
----------------------------------------------------------------------
  Total NPUs:          4
  Min Wall Time:       4,932,010 cycles
  Max Wall Time:       4,932,010 cycles
  Avg Wall Time:       4,932,010 cycles
  Variance:            0.00%
  Status:              [OK] Excellent load balance (<1% variance)
```

### 2. 对比多个配置

```bash
# 对比 DP4 和 TP4
python analyze_log.py \
    output/dp4_h100/logs/simulation_log_dp4_h100.txt \
    output/tp4_h100/logs/simulation_log_tp4_h100.txt \
    --compare

# 自定义输出路径
python analyze_log.py \
    output/dp4_h100/logs/simulation_log_dp4_h100.txt \
    output/tp4_h100/logs/simulation_log_tp4_h100.txt \
    --compare --output comparison_dp_vs_tp.txt
```

**对比输出示例：**

```
======================================================================
ASTRA-sim Multi-Configuration Comparison
======================================================================

[OK] Loaded: output/dp4_h100/logs/simulation_log_dp4_h100.txt
[OK] Loaded: output/tp4_h100/logs/simulation_log_tp4_h100.txt

======================================================================
Comparison Table
======================================================================
Metric                            Config 1        Config 2
----------------------------------------------------------------------
Wall Time (cycles)               3,287,100       4,932,010
GPU Time (cycles)                1,870,980       3,154,090
Comm Time (cycles)               1,416,120       4,846,452
GPU %                                56.90           63.95
Comm %                               43.08           98.27
Compute Util %                       85.49           89.09
Bottleneck            COMPUTE_BOUND  SEVERE_COMM_BOUND

======================================================================
Speedup Analysis (vs Config 1)
======================================================================
  Config 1: simulation_log_dp4_h100.txt
    Baseline: 3,287,100 cycles
  Config 2: simulation_log_tp4_h100.txt
    Speedup: 0.67x
    Time saved: -50.0%
```

### 3. 命令行参数

```bash
python analyze_log.py --help
```

**参数说明：**

| 参数 | 简写 | 说明 |
|------|------|------|
| `log_files` | - | 日志文件路径（必需，可多个） |
| `--compare` | `-c` | 对比模式（需要2个以上日志文件） |
| `--output` | `-o` | 指定输出文件路径 |
| `--no-save` | - | 不保存文件，只输出到终端 |

## 🔍 如何解读分析结果

### 1. 性能指标解读

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **Wall Time** | 总执行时间 | 越小越好 |
| **GPU %** | 计算时间占比 | 高表示计算密集 |
| **Comm %** | 通信时间占比 | 低表示通信开销小 |
| **Overlap %** | 计算通信重叠 | 越高越好（异步执行） |

**注意：** Comm % 可能 > 100%，因为可能有重叠部分

### 2. 瓶颈判断标准

```python
瓶颈分类：
- Comm % > 90%  →  "SEVERE_COMM_BOUND" (严重通信瓶颈)
- Comm % > 70%  →  "COMM_BOUND" (通信瓶颈)
- GPU % > 60%   →  "COMPUTE_BOUND" (计算瓶颈)
- 其他          →  "BALANCED" (平衡)
```

### 3. 优化建议

根据瓶颈类型：

**通信瓶颈 (Comm-bound):**
- ✓ 增加网络带宽
- ✓ 优化集合通信算法
- ✓ 减少通信频率
- ✓ 增加计算/通信重叠

**计算瓶颈 (Compute-bound):**
- ✓ 优化算子实现
- ✓ 提高 GPU 利用率
- ✓ 增加批处理大小

## 📊 实际案例

### 案例：TP4 vs DP4 分析

从你的日志分析可以看出：

**TP4 (Tensor Parallel = 4):**
- Wall Time: 4,932,010 cycles
- Comm %: 98.27% → **严重通信瓶颈**
- Compute Util: 89.09%
- 结论：张量并行引入大量 AllGather/AllReduce

**DP4 (Data Parallel = 4):**
- Wall Time: 3,287,100 cycles (快 33%)
- Comm %: 43.08% → **计算为主**
- Compute Util: 85.49%
- 结论：数据并行通信开销小，性能更好

**建议：**
对于这个工作负载和规模，DP4 性能优于 TP4

## 🛠️ 脚本核心逻辑

```python
# 1. 正则表达式提取
r'sys\[0\].*Wall time: (\d+)'           # 提取 Wall Time
r'sys\[0\].*GPU time: (\d+)'            # 提取 GPU Time
r'sys\[0\].*Comm time: (\d+)'           # 提取 Comm Time
# ... 更多模式

# 2. 计算衍生指标
gpu_pct = (gpu_time / wall_time) * 100

# 3. 瓶颈识别
if comm_pct > 90:
    return "SEVERE_COMM_BOUND"
```

## 总结

`analyze_log.py` 的设计是**合理且实用的**：

✅ **优点：**
- 自动化提取关键指标
- 直观的可视化输出
- 支持多配置对比
- 智能瓶颈分析

⚠ **局限性：**
- 只提取汇总统计，没有逐层/逐算子分析
- 没有通信模式细节
- 假设所有NPU负载均衡（需验证）

**推荐使用场景：**
1. 快速对比不同配置的性能
2. 识别主要性能瓶颈
3. 生成性能报告
4. CI/CD 性能回归测试

