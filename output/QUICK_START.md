# Quick Start Guide

## ? 快速查看结果

### 1?? 查看性能对比（推荐从这里开始）

```bash
cat comparison/COMPARISON_SUMMARY.md
```

**关键结论**: Megatron TP=8 比 Mesh2D TP=24 快 **4.76×**

---

### 2?? 查看 Mesh2D TP=24 详细结果

```bash
# 查看配置信息
cat mesh2d_tp24/CONFIG_INFO.md

# 查看性能分析
cat mesh2d_tp24/analysis/simulation_summary.md

# 查看原始日志
less mesh2d_tp24/logs/simulation_log.txt

# 查看 workload 文件
ls mesh2d_tp24/workloads/
```

**关键指标**:
- Wall Time: 35.3M cycles
- Communication: 96.9% (严重通信瓶颈 ??)
- Compute Utilization: 13.69%

---

### 3?? 查看 Megatron TP=8 详细结果

```bash
# 查看配置信息
cat megatron_tp8/CONFIG_INFO.md

# 查看性能分析
cat megatron_tp8/analysis/simulation_megatron_tp8_summary.md

# 查看原始日志
less megatron_tp8/logs/simulation_log_megatron_tp8.txt

# 查看 workload 文件
ls megatron_tp8/workloads/
```

**关键指标**:
- Wall Time: 7.4M cycles (**4.76× faster** ?)
- Communication: 82.2% (有所改善)
- Compute Utilization: 34.33% (**2.5× better** ?)

---

## ? 快速对比命令

### 提取关键指标对比

```bash
# Mesh2D TP=24 结果
echo "=== Mesh2D TP=24 ==="
grep "sys\[0\]" mesh2d_tp24/logs/simulation_log.txt | grep -E "(Wall time|Comm time|Compute utilization)"

echo ""
echo "=== Megatron TP=8 ==="
grep "sys\[0\]" megatron_tp8/logs/simulation_log_megatron_tp8.txt | grep -E "(Wall time|Comm time|Compute utilization)"
```

### 计算加速比

```bash
# 使用 Python 快速计算
python3 << EOF
mesh2d_time = 35334744
megatron_time = 7424768
speedup = mesh2d_time / megatron_time
print(f"Speedup: {speedup:.2f}x")
print(f"Time saved: {(1 - megatron_time/mesh2d_time)*100:.1f}%")
EOF
```

---

## ? 目录结构一览

```
output/
├── README.md                   # 完整说明文档
├── QUICK_START.md             # 这个文件（快速开始）
│
├── comparison/                 # 对比分析
│   └── COMPARISON_SUMMARY.md  # ? 先看这个！
│
├── mesh2d_tp24/               # Mesh2D 配置结果
│   ├── CONFIG_INFO.md         # 配置详情
│   ├── logs/
│   │   └── simulation_log.txt
│   ├── workloads/
│   │   ├── attention_tp24.{0-23}.et    (24 files)
│   │   └── attention_tp24.json
│   └── analysis/
│       └── simulation_summary.md
│
└── megatron_tp8/              # Megatron 配置结果
    ├── CONFIG_INFO.md         # 配置详情
    ├── logs/
    │   └── simulation_log_megatron_tp8.txt
    ├── workloads/
    │   ├── attention_tp8.{0-7}.et      (8 files)
    │   └── attention_tp8.json
    └── analysis/
        └── simulation_megatron_tp8_summary.md
```

---

## ? 推荐阅读顺序

1. **`comparison/COMPARISON_SUMMARY.md`** ?
   - 快速了解两种配置的性能差异
   - 包含关键指标对比表格
   - 约 5 分钟阅读

2. **`mesh2d_tp24/CONFIG_INFO.md`**
   - 了解 Mesh2D TP=24 的配置
   - 约 2 分钟阅读

3. **`megatron_tp8/CONFIG_INFO.md`**
   - 了解 Megatron TP=8 的配置
   - 约 2 分钟阅读

4. **详细分析** (可选)
   - `mesh2d_tp24/analysis/simulation_summary.md`
   - `megatron_tp8/analysis/simulation_megatron_tp8_summary.md`
   - 完整的性能分析和优化建议

5. **原始日志** (调试用)
   - `mesh2d_tp24/logs/simulation_log.txt`
   - `megatron_tp8/logs/simulation_log_megatron_tp8.txt`

---

## ? 关键发现

### ? Megatron TP=8 的优势

1. **4.76× 更快的执行时间**
   - Mesh2D: 35.3M cycles
   - Megatron: 7.4M cycles

2. **5.61× 更少的通信时间**
   - 得益于 NVSwitch 单跳通信
   - 400 GB/s vs 100 GB/s 带宽

3. **2.5× 更高的计算利用率**
   - 34.33% vs 13.69%
   - GPU 空闲时间更少

4. **更好的负载均衡**
   - 8 heads/GPU (perfect) vs 2.67 heads/GPU (uneven)

### ?? 仍然存在的瓶颈

两种配置都是**通信主导**的：
- Mesh2D: 96.9% 通信
- Megatron: 82.2% 通信

**优化方向**:
- 增加 batch size
- 增加 transformer 层数
- 使用混合精度减少通信量

---

## ? 重现这些结果

完整的重现步骤请参考 `README.md`。

简短版本：
```bash
# 回到项目根目录
cd ..

# 运行 Mesh2D TP=24
bash examples/run_scripts/analytical/congestion_aware/Mesh2D_allgather_24npus_6x4.sh

# 运行 Megatron TP=8
bash examples/run_scripts/analytical/congestion_aware/Megatron_TP8_attention.sh
```

---

## ? 问题反馈

如果有任何问题或需要进一步分析，请参考：
- ASTRA-sim 文档: https://astra-sim.github.io/
- 完整的 README: `README.md`

---

**Last Updated**: 2025-12-18

