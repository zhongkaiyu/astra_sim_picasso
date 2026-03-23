# hmp_reo_new 各阶段 HBM 读取 & D2D 传输 — 原始数据

**数据来源**: 严格从以下文件读取
- `reports/hybrid/gqa_hybrid_merged_80T_split4_bw1500.json` (`strategies.hmp_reo_new.data`)
- `reports/roofline/gqa_roofline_bs1_80T_bw1500.json` (`strategies.hmp_reo_new.data`)

**单位说明**:
- HBM: roofline 中为 `elems` (elements)；merge 将其作为 `hbm_read_bytes_per_cube` 使用（1 elem = 1 byte 约定）
- D2D: `bytes` (所有 cube 上 D2D 链路的传输总量)

---

## 表 1: HBM 读取 — 各阶段 element 数量 (per cube)

| Seq | Proj_QKV (weight+act) | Attention (K+V cache) | Proj_O (weight+act) | HBM total per_cube |
|-----|-----------------------|------------------------|---------------------|--------------------|
| 1K  | 2,363,968             | 65,536                 | 2,103,296           | 4,532,800          |
| 2K  | 2,363,968             | 131,072                | 2,103,296           | 4,598,336          |
| 4K  | 2,363,968             | 262,144                | 2,103,296           | 4,729,408          |
| 8K  | 2,363,968             | 524,288                | 2,103,296           | 4,991,552          |
| 16K | 2,363,968             | 1,048,576              | 2,103,296           | 5,515,840          |
| 32K | 2,363,968             | 2,097,152              | 2,103,296           | 6,564,416          |
| 64K | 2,363,968             | 4,194,304              | 2,103,296           | 8,661,568          |
| 128K| 2,363,968             | 8,388,608              | 2,103,296           | 12,855,872         |
| 256K| 2,363,968             | 16,777,216             | 2,103,296           | 21,244,480         |
| 512K| 2,363,968             | 33,554,432             | 2,103,296           | 38,021,696         |
| 1M  | 2,363,968             | 67,108,864             | 2,103,296           | 71,576,128         |

**说明**:
- Proj_QKV / Proj_O 不随 seq 变化（权重 + 激活固定）
- Attention 的 KV cache = 2 × concat_k = 2 × concat_v，随 seq 线性增长
- concat_k = concat_v = `cache_seq × Hkv × dk`，cache_seq = ceil(seq / tp_s) = ceil(seq/4)

---

## 表 2: D2D 传输 — 各阶段 bytes (all cubes)

| Seq | qkv_allgather | attn_rs | final_reduce | D2D total |
|-----|---------------|---------|--------------|-----------|
| 全 seq | 27,648 B | 6,144 B | 61,440 B | **95,232 B** |

**说明**: hmp_reo_new 的 D2D 与 seq 无关，通信模式固定：
- `qkv_allgather`: tp_s=4 个组，ring AG
- `attn_rs`: tp_s=4 个组，ring RS (block2x2 1-hop)
- `final_reduce`: 16 NPU tree-reduce

---

## 表 3: HBM 明细 (roofline 字段)

| 阶段 | 字段 | 值 (seq=1024 示例) |
|------|------|---------------------|
| Proj_QKV | weight_elems | 2,359,296 |
| Proj_QKV | activation_elems | 4,672 |
| Attention | concat_k.elems | 32,768 |
| Attention | concat_v.elems | 32,768 |
| Proj_O | weight_elems | 2,097,152 |
| Proj_O | activation_elems | 6,144 |
