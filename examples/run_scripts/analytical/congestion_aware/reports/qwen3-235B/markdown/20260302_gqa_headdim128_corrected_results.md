# GQA-Only 修正报告：HeadDim=128 仿真结果

> **生成时间**: 2026-03-02 20:40 UTC
> **修正内容**: 修复 `bulk_run_configs.py` 中 `head_dim` 参数传递缺失问题，使用正确的 HeadDim=128 重新生成所有 trace 并重跑仿真
> **设备**: HBM4 | **物理拓扑**: Mesh2D 4×4 | **逻辑路由**: OneRing
> **模型参数**: Qwen (head=64, kvhead=4, dmodel=4096, head_dim=128, batch=64, seq=1024)

---

## 1. 问题描述

### 1.1 Bug: `bulk_run_configs.py` 不传递 `head_dim`

`build_trace_cmd()` 中的 `flag_keys` 列表缺少 `"head_dim"`，导致 trace config JSON 中的 `"head_dim": 128` 从未被传递给 `main.py`。trace 始终使用默认值 `dmodel // head = 4096 // 64 = 64`。

**修复** (2026-03-02): 在 `flag_keys` 中添加 `"head_dim"` 和 `"rank_remap"`。

### 1.2 CSV 计算图 qkv op_attr 表达式变更

从 `2*Dmodel/Head` 迁移到 `HeadDim` 时，GQA kernel 的 Flash Attention CUSTOM op 表达式也被修改：

| 版本 | qkv op_attr 表达式 | 数值 (tp_h=4, tp_s=4) |
|------|-------------------|---:|
| **旧 CSV** | `Batch/dp * 1 * 2*Dmodel/Head * Head/tp_h * 3` | 393,216 |
| **新 CSV** | `Batch/dp * Seq/tp_s * HeadDim * Head/tp_h * 3` | 100,663,296 |

差异来源：新版本加入了 `Seq/tp_s` (KV cache 长度) 因子，使 qkv FLOPs 增大 ~256×。

### 1.3 Trace 节点对比验证

通过读取 `.et` protobuf 文件逐节点对比旧 trace (Feb 27 block2x2 未覆盖版) 与新 HeadDim=128 trace：

| 节点 | 旧 trace (Feb 27) | 新 HeadDim=128 | 状态 |
|------|---:|---:|------|
| q matmul (ops) | 536,870,912 | 536,870,912 | 一致 |
| kv matmul (ops) | 67,108,864 | 67,108,864 | 一致 |
| o matmul (ops) | 536,870,912 | 536,870,912 | 一致 |
| k concat (tsz) | 2,103,296 | 2,103,296 | 一致 |
| v concat (tsz) | 2,103,296 | 2,103,296 | 一致 |
| k1/v1 comm (csz) | 2,103,296 | 2,103,296 | 一致 |
| **qkv CUSTOM (ops)** | **393,216** | **100,663,296** | **差异 (op_attr 变更)** |

**结论**: 旧缓存结果（HMP wall=56,635）与新 HeadDim=128 结果（HMP wall=59,938）的差异**全部**来自 qkv CUSTOM op_attr 表达式变更，而非 HeadDim 符号替换。

---

## 2. 修正后的 GQA-Only 单层结果

### 2.1 HBM4 Real Mesh2D 4×4 OneRing (HeadDim=128)

| 策略 | Wall (ns) | GPU (ns) | Comm (ns) | vs TP16 原版 |
|------|---:|---:|---:|---:|
| **TP16 v2** (KVHead 切分, tp_h=4 tp_s=4) | **37,063** | **21,253** | **15,810** | — |
| HMP block2x2 (tp_h=4, tp_s=4) | 54,418 | 43,069 | 12,190 | +46.8% |
| HMP (tp_h=4, tp_s=4) | 59,938 | 43,069 | 17,710 | +61.7% |
| Baseline block2x2 (tp_h=4, tp_s=4) | 59,261 | 29,647 | 30,455 | +59.9% |
| Baseline (tp_h=4, tp_s=4) | 65,993 | 29,647 | 37,187 | +78.1% |

### 2.2 Block2x2 优化效果 (HeadDim=128)

| 策略 | 原版 wall | block2x2 wall | Wall 改善 | Comm 改善 |
|------|---:|---:|---:|---:|
| HMP | 59,938 | 54,418 | **-9.2%** | -31.2% |
| Baseline | 65,993 | 59,261 | **-10.2%** | -18.1% |

### 2.3 与旧结果对比 (HeadDim=64 → 128)

| 策略 | 旧值 (HeadDim=64) | 新值 (HeadDim=128) | 变化 |
|------|---:|---:|---:|
| HMP wall | 37,653 | 59,938 | +59.2% |
| HMP gpu | 21,531 | 43,069 | +100.0% |
| Baseline wall | 50,419 | 65,993 | +30.9% |
| Baseline gpu | 14,820 | 29,647 | +100.0% |
| TP16 v2 wall | 26,433 | 37,063 | +40.2% |
| TP16 v2 gpu | 10,623 | 21,253 | +100.1% |

GPU 时间约为 2 倍是预期的——HeadDim 从 64→128，所有涉及 HeadDim 的矩阵乘法 FLOPs 翻倍。

---

## 3. 历史数据时间线

| 时间 | 事件 | 数据状态 |
|------|------|---------|
| 2026-02-27 12:15 | 初始 GQA trace 生成和仿真 | CSVs 使用混合表达式，HeadDim 默认=64 |
| 2026-02-27 19:24 | 添加 block2x2 配置 | 同上，block2x2 trace 未覆盖 |
| 2026-03-02 18:47 | 创建 TP16 v2 (KVHead 切分版) | 直接调用 main.py，head_dim=128 正确 |
| 2026-03-02 19:50 | 添加 head_dim=128 到 trace configs | bulk_run_configs.py 未传递 → trace 仍 HeadDim=64 |
| 2026-03-02 19:56 | --trace-force 重新生成 HMP/Baseline | 使用修改后的 CSVs + HeadDim=64 (错误) |
| **2026-03-02 20:34** | **修复 bulk_run_configs.py** | **添加 head_dim 到 flag_keys** |
| **2026-03-02 20:36** | **重新生成所有 trace (HeadDim=128)** | **正确结果** |
| **2026-03-02 20:39** | **重新生成 block2x2 trace (HeadDim=128)** | **正确结果** |

---

## 4. 修复的代码变更

### `bulk_run_configs.py` — `build_trace_cmd()` flag_keys

```python
# 修复前
flag_keys = [
    "dp", "tp", "pp", "sp", "ep", ...,
    "tp_h", "tp_s",
]

# 修复后
flag_keys = [
    "dp", "tp", "pp", "sp", "ep", ...,
    "tp_h", "tp_s",
    "head_dim",      # ← 新增
    "rank_remap",    # ← 新增
]
```

---

## 5. 验证方法

### 5.1 验证 trace 节点值

```bash
cd symbolic_tensor_graph_picasso
python3 -c "
import sys; sys.path.insert(0, 'symbolic_tensor_graph/chakra/backends/chakra_00_4_backend')
from et_def.et_def_pb2 import Node, GlobalMetadata
from protolib import openFileRd, decodeMessage
f = openFileRd('et_trace/generated_gqa_hmp_fwd_tph4_tps4/gqa_hmp_fwd_tph4_tps4.0.et')
gm = GlobalMetadata(); decodeMessage(f, gm)
while True:
    n = Node()
    if not decodeMessage(f, n): break
    ops = next((a.int64_val for a in n.attr if a.name=='num_ops'), 0)
    tsz = next((a.uint64_val for a in n.attr if a.name=='tensor_size'), 0)
    print(f'{n.name:55s} ops={ops:>15,} tsz={tsz:>12,}')
"
```

期望 `mha.q@0_COMP` 的 ops=536,870,912 (HeadDim=128)，而非 268,435,456 (HeadDim=64)。

### 5.2 快速验证公式

- q matmul ops = Batch/dp × 1 × Dmodel × HeadDim × Head/tp_h = 64 × 4096 × **128** × 16 = **536,870,912** ✓
- qkv CUSTOM ops = Batch/dp × Seq/tp_s × HeadDim × Head/tp_h × 3 = 64 × 256 × **128** × 16 × 3 = **100,663,296** ✓

---

## 附件

| 文件 | 说明 |
|------|------|
| [`gqa_analysis_hbm4_onering.md`](gqa_analysis_hbm4_onering.md) | 原始 Ring 拓扑分析报告 (HeadDim=64, 已过时) |
| [`ring_vs_mesh2d_gqa_comparison.md`](ring_vs_mesh2d_gqa_comparison.md) | Ring vs Mesh2D 拓扑对比 (HeadDim=64, 已过时) |
| `../cache_db_v131/results.sqlite` | 完整仿真结果数据库 |
