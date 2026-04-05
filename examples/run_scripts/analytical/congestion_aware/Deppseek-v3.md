# DeepSeek V3 Multi-head Latent Attention (MLA) 调研报告

## 1. 背景与动机

Transformer 推理（decode）阶段的核心瓶颈是 **KV Cache** 的显存占用。对于标准 Multi-Head Attention (MHA)，KV Cache 大小为：

$$
\text{KV Cache}_{\text{MHA}} = 2 \times n_{\text{layers}} \times s \times n_h \times d_h
$$

其中 $s$ 为序列长度，$n_h$ 为注意力头数，$d_h$ 为每头维度。以 DeepSeek-V3 为例（$n_h=128, d_h=128$），每 token 每层需要缓存 $2 \times 128 \times 128 = 32768$ 个元素，随序列长度线性增长，成为长序列推理的主要瓶颈。

GQA (Grouped Query Attention) 通过让多个 Query 头共享 KV 头来压缩缓存，但本质上是有损的——更少的 KV 头意味着更少的表达能力。

**MLA (Multi-head Latent Attention)** 由 DeepSeek-V2 提出，DeepSeek-V3 延续使用，采用了根本不同的方案：**对 Key 和 Value 进行联合低秩压缩**，并通过**矩阵吸收 (Absorption)** 技巧使推理时完全无需解压，在压缩空间中直接完成注意力计算。

---

## 2. MLA 核心架构

### 2.1 符号定义

| 符号 | 含义 | DeepSeek-V3 数值 |
|------|------|------------------|
| $d_{\text{model}}$ | 隐藏层维度 | 7168 |
| $n_h$ | 注意力头数 | 128 |
| $d_h$ | 每头维度 | 128 |
| $d_c$ | KV 压缩潜变量维度 | 512 |
| $d_c'$ | Query 压缩维度 | 1536 |
| $d_r$ | RoPE 维度 | 64 |
| $n_{\text{layers}}$ | 层数 | 61 |

### 2.2 KV 联合低秩压缩

MLA 不再分别投影 Key 和 Value，而是将二者**联合压缩**为一个低维潜变量：

$$
\mathbf{c}_t^{KV} = W_{DKV} \cdot \mathbf{h}_t \quad \in \mathbb{R}^{d_c}
$$

其中 $W_{DKV} \in \mathbb{R}^{d_c \times d_{\text{model}}}$ 为下投影矩阵。

概念上，完整的 Key 和 Value 可通过上投影恢复：

$$
\mathbf{k}_t^C = W_{UK} \cdot \mathbf{c}_t^{KV} \quad \in \mathbb{R}^{n_h \cdot d_h}
$$

$$
\mathbf{v}_t = W_{UV} \cdot \mathbf{c}_t^{KV} \quad \in \mathbb{R}^{n_h \cdot d_h}
$$

但关键点在于：**推理时永远不需要真正执行这两个上投影**（见第 3 节矩阵吸收）。

### 2.3 Query 压缩

Query 同样采用低秩压缩（用于减少参数量，非为 KV Cache）：

$$
\mathbf{c}_t^Q = W_{DQ} \cdot \mathbf{h}_t \quad \in \mathbb{R}^{d_c'}
$$

$$
\mathbf{q}_t^C = W_{UQ} \cdot \mathbf{c}_t^Q \quad \in \mathbb{R}^{n_h \cdot d_h}
$$

### 2.4 RoPE 的解耦处理

**问题**：RoPE 是位置相关的逐元素旋转变换。如果将 RoPE 应用到 $\mathbf{k}_t^C = W_{UK} \cdot \mathbf{c}_t^{KV}$ 上，位置相关的旋转会与矩阵乘法交织，**破坏矩阵吸收的数学基础**。

**MLA 的解决方案**：将 Key 和 Query 解耦为两部分：

1. **内容部分** (Content)：不带 RoPE，参与矩阵吸收
2. **位置部分** (Positional)：携带 RoPE，单独处理

具体地，MLA 额外生成一个小的 RoPE 分量：

$$
\mathbf{k}_t^R = \text{RoPE}(W_{KR} \cdot \mathbf{h}_t, \, \text{pos}_t) \quad \in \mathbb{R}^{d_r}
$$

$$
\mathbf{q}_t^{R,(i)} = \text{RoPE}(W_{QR}^{(i)} \cdot \mathbf{c}_t^Q, \, \text{pos}_t) \quad \in \mathbb{R}^{d_r}
$$

最终每头的完整 Query 和 Key 由拼接构成：

$$
\mathbf{q}_t^{(i)} = [\, \mathbf{q}_t^{C,(i)} \;;\; \mathbf{q}_t^{R,(i)} \,] \quad \in \mathbb{R}^{d_h + d_r}
$$

$$
\mathbf{k}_t^{(i)} = [\, \mathbf{k}_t^{C,(i)} \;;\; \mathbf{k}_t^R \,] \quad \in \mathbb{R}^{d_h + d_r}
$$

### 2.5 KV Cache 的实际存储

每 token 每层实际缓存的内容：

$$
\text{KV Cache} = \mathbf{c}_t^{KV} \in \mathbb{R}^{d_c} \;+\; \mathbf{k}_t^R \in \mathbb{R}^{d_r}
$$

$$
\text{每 token 每层} = d_c + d_r = 512 + 64 = 576 \text{ 个元素}
$$

**相比 MHA 的压缩比**：$32768 / 576 \approx \mathbf{56.9\times}$

---

## 3. 矩阵吸收 (Weight Absorption) —— 核心创新

### 3.1 问题：朴素解压的开销

如果在推理时先解压再计算注意力：
1. 从 Cache 加载 $\mathbf{c}_t^{KV}$
2. 计算 $\mathbf{k}_t^C = W_{UK} \cdot \mathbf{c}_t^{KV}$（解压 Key）
3. 计算 $\mathbf{v}_t = W_{UV} \cdot \mathbf{c}_t^{KV}$（解压 Value）
4. 执行标准注意力

这会在每个 decode step 对所有缓存 token 执行大矩阵乘法，完全抵消了压缩的收益。

### 3.2 解决方案：将上投影吸收进 Q 和 O

**核心数学恒等式**：利用矩阵乘法的**结合律**，改变括号化方式。

#### 3.2.1 吸收 $W_{UK}$ 到 Query 投影中

第 $i$ 头在位置 $t$（query）和位置 $j$（key）之间的内容注意力分数：

$$
\text{score}_{\text{content}} = (\mathbf{q}_t^{C,(i)})^\top \cdot \mathbf{k}_j^{C,(i)}
$$

展开：

$$
= (W_{UQ}^{(i)} \cdot \mathbf{c}_t^Q)^\top \cdot (W_{UK}^{(i)} \cdot \mathbf{c}_j^{KV})
$$

$$
= (\mathbf{c}_t^Q)^\top \cdot \underbrace{(W_{UQ}^{(i)})^\top \cdot W_{UK}^{(i)}}_{W_{Q,\text{abs}}^{(i)}} \cdot \mathbf{c}_j^{KV}
$$

定义**吸收后的 Query 权重**：

$$
W_{Q,\text{abs}}^{(i)} = (W_{UQ}^{(i)})^\top \cdot W_{UK}^{(i)} \quad \in \mathbb{R}^{d_c' \times d_c}
$$

或者等价地，计算"吸收后的 Query 向量"：

$$
\mathbf{q}_{t,\text{abs}}^{(i)} = (W_{UK}^{(i)})^\top \cdot W_{UQ}^{(i)} \cdot \mathbf{c}_t^Q \quad \in \mathbb{R}^{d_c}
$$

此时注意力分数变为：

$$
\text{score}_{\text{content}} = (\mathbf{q}_{t,\text{abs}}^{(i)})^\top \cdot \mathbf{c}_j^{KV}
$$

**直接用压缩后的 $\mathbf{c}_j^{KV}$ 计算，无需解压 Key！**

#### 3.2.2 吸收 $W_{UV}$ 到 Output 投影中

注意力加权求和：

$$
\mathbf{o}_t^{(i)} = \sum_j \alpha_{t,j} \cdot \mathbf{v}_j^{(i)} = \sum_j \alpha_{t,j} \cdot W_{UV}^{(i)} \cdot \mathbf{c}_j^{KV}
$$

利用线性性提出 $W_{UV}^{(i)}$：

$$
= W_{UV}^{(i)} \cdot \underbrace{\sum_j \alpha_{t,j} \cdot \mathbf{c}_j^{KV}}_{\mathbf{o}_{t,\text{comp}}^{(i)} \in \mathbb{R}^{d_c}}
$$

先在压缩空间中完成加权求和，再投影。进一步，$W_{UV}^{(i)}$ 可与输出投影 $W_O^{(i)}$ 合并：

$$
W_{O,\text{abs}}^{(i)} = W_O^{(i)} \cdot W_{UV}^{(i)} \quad \in \mathbb{R}^{d_{\text{model}} \times d_c}
$$

---

## 4. Decode 阶段完整推理流程

### 4.1 预计算（模型加载时，一次性）

```
对每个注意力头 i:
    W_Q_abs[i] = (W_UK[i])^T · W_UQ[i]     ∈ R^{d_c × d_c'}
    W_O_abs[i] = W_O[i] · W_UV[i]           ∈ R^{d_model × d_c}
```

### 4.2 每 Token Decode 步骤

```
输入: h_t ∈ R^{d_model}（当前 token 的隐藏状态）
      Cache: {c_j^KV, k_j^R} for j = 1, ..., t-1

Step 1: KV 压缩与缓存
    c_t^KV = W_DKV · h_t                              ∈ R^{d_c}     (512维)
    k_t^R  = RoPE(W_KR · h_t, pos_t)                  ∈ R^{d_r}     (64维)
    → 将 c_t^KV 和 k_t^R 追加到 Cache

Step 2: 计算 Query（使用吸收后的权重）
    c_t^Q = W_DQ · h_t                                 ∈ R^{d_c'}    (1536维)
    对每个头 i:
        q_t_abs[i] = W_Q_abs[i] · c_t^Q               ∈ R^{d_c}     (512维, 内容query)
        q_t^R[i]   = RoPE(W_QR[i] · c_t^Q, pos_t)     ∈ R^{d_r}     (64维, 位置query)

Step 3: 计算注意力分数（直接在压缩空间）
    对每个头 i, 对每个缓存位置 j:
        score[i]_{t,j} = (q_t_abs[i])^T · c_j^KV      // 内容部分，512维点积
                       + (q_t^R[i])^T · k_j^R          // 位置部分，64维点积
        score[i]_{t,j} /= sqrt(d_h + d_r)

Step 4: Softmax
    α[i]_{t,j} = softmax_j(score[i]_{t,j})

Step 5: 在压缩空间聚合（不解压 Value！）
    o_comp[i] = Σ_j α[i]_{t,j} · c_j^KV               ∈ R^{d_c}     (512维)

Step 6: 使用吸收后的输出投影
    o_t = Σ_i W_O_abs[i] · o_comp[i]                   ∈ R^{d_model} (7168维)
```

**核心要点：整个注意力计算过程中，完整维度的 Key 和 Value 从未被显式构造。所有操作都在 $d_c = 512$ 维的压缩空间中完成。**

---

## 5. MHA vs GQA vs MLA 全面对比

### 5.1 KV Cache 大小对比

| 方法 | 每 token 每层缓存大小 | 以 DeepSeek-V3 规模计 | 相对 MHA 压缩比 |
|------|----------------------|----------------------|----------------|
| **MHA** | $2 \times n_h \times d_h$ | $2 \times 128 \times 128 = 32768$ | 1x |
| **GQA** (g 组) | $2 \times g \times d_h$ | $2 \times 8 \times 128 = 2048$ (8组) | 16x |
| **MQA** (1 组) | $2 \times d_h$ | $2 \times 128 = 256$ | 128x |
| **MLA** | $d_c + d_r$ | $512 + 64 = 576$ | **56.9x** |

### 5.2 核心差异对比

| 维度 | MHA | GQA | MLA |
|------|-----|-----|-----|
| **压缩原理** | 无压缩 | KV 头共享（有损） | 学习到的低秩联合压缩 |
| **信息损失** | 无 | 共享头牺牲表达能力 | 低秩瓶颈保留主要信息 |
| **每头独立性** | 每头独立 KV | 多个 Q 头共享 KV 头 | 每头通过独立 $W_{UK}^{(i)}, W_{UV}^{(i)}$ 从共享潜变量提取不同信息 |
| **推理时额外计算** | 标准 QK 和 AV 乘法 | 同 MHA（更少的头） | 吸收后无解压，但吸收后的权重矩阵更大 |
| **模型质量** | 基准 | 随组数减少而下降 | 持平或优于 MHA |
| **位置编码** | 直接应用 RoPE | 直接应用 RoPE | RoPE 解耦到独立分量 |
| **训练开销** | 基准 | 同 MHA 或更少 | 略多（压缩/解压层） |

### 5.3 计算流程对比

#### MHA Decode 流程:
```
Q = h · W_Q                    # [1, n_h × d_h]
K_new = h · W_K                # [1, n_h × d_h], 追加到 KV Cache
V_new = h · W_V                # [1, n_h × d_h], 追加到 KV Cache
scores = Q · K_cache^T         # [n_h, 1, seq_len]
attn = softmax(scores)
output = attn · V_cache        # [n_h, 1, d_h]
output = concat → W_O          # [1, d_model]
```

#### GQA Decode 流程:
```
Q = h · W_Q                    # [1, n_h × d_h]
K_new = h · W_K                # [1, g × d_h], g << n_h, 追加到 KV Cache
V_new = h · W_V                # [1, g × d_h], 追加到 KV Cache
# 每组 KV 头被 n_h/g 个 Q 头共享
scores = Q · K_cache^T         # 广播 KV 头到对应 Q 头组
attn = softmax(scores)
output = attn · V_cache
output = concat → W_O
```

#### MLA Decode 流程:
```
c_KV = h · W_DKV               # [1, d_c=512], 追加到 Cache（压缩态）
k_R  = RoPE(h · W_KR)          # [1, d_r=64],  追加到 Cache
c_Q  = h · W_DQ                # [1, d_c'=1536]
q_abs = c_Q · W_Q_abs          # [n_h, d_c=512], 已吸收 W_UK
q_R  = RoPE(c_Q · W_QR)        # [n_h, d_r=64]
scores = q_abs · c_KV_cache^T + q_R · k_R_cache^T   # 压缩空间计算
attn = softmax(scores)
o_comp = attn · c_KV_cache     # [n_h, d_c=512], 压缩空间聚合
output = o_comp · W_O_abs      # [1, d_model], 已吸收 W_UV
```

### 5.4 为什么 MLA 优于 GQA

1. **表达能力保留**：GQA 通过共享 KV 头来压缩，每组内的 Q 头看到完全相同的 KV 信息。MLA 虽然共享同一个压缩潜变量 $\mathbf{c}^{KV}$，但每个头通过独立的上投影矩阵 $W_{UK}^{(i)}, W_{UV}^{(i)}$ 可以提取不同的信息，表达能力更强。

2. **端到端学习的压缩**：低秩压缩是在训练中学习到的，模型自动学会保留注意力计算中最重要的信息。

3. **正则化效果**：低秩瓶颈对注意力机制起到了隐式正则化的作用。

4. **实验验证**：DeepSeek-V2 的实验表明，MLA 在 KV Cache 大小与 GQA-2 组（仅 2 个 KV 头）相当的情况下，性能达到甚至超过完整 MHA。

---

## 6. 吸收后权重矩阵的代价

矩阵吸收并非没有代价。吸收后的权重矩阵更大：

| 权重矩阵 | 维度 | 说明 |
|----------|------|------|
| $W_{Q,\text{abs}}^{(i)}$ | $d_c \times d_c'$ = $512 \times 1536$ | 吸收 $W_{UK}$ 后 |
| $W_{O,\text{abs}}^{(i)}$ | $d_{\text{model}} \times d_c$ = $7168 \times 512$ | 吸收 $W_{UV}$ 后 |

但这是**固定开销**（模型参数），不随序列长度增长。而 KV Cache 随序列长度线性增长。对于长序列推理场景，MLA 的收益远大于这个固定开销。

---

## 7. 总结

### MLA 的核心思想（一句话）

> 通过低秩联合压缩 KV，仅缓存压缩潜变量；利用矩阵乘法结合律，将解压矩阵吸收到 Query 和 Output 投影中，使推理时永远不需要解压，整个注意力计算在压缩空间中完成。

### MLA 设计的三个关键洞察

1. **联合压缩**：Key 和 Value 共享同一个压缩潜变量 $\mathbf{c}^{KV}$，只需缓存一份
2. **矩阵吸收**：$W_{UK}$ 吸收到 $W_{UQ}$ 中形成吸收后的 Query 投影；$W_{UV}$ 吸收到 $W_O$ 中形成吸收后的 Output 投影
3. **RoPE 解耦**：位置编码通过独立的小维度分量处理，不干扰矩阵吸收

### 最终效果

- KV Cache 压缩约 **57 倍**（相对 MHA）
- 模型质量**持平或优于** MHA
- 推理速度和吞吐量大幅提升（KV Cache 减小 → 更大 batch size → 更高 GPU 利用率）

---

## 附录 A：DeepSeek-V3 MLA Decode 逐步计算流程（标注全部维度）

> 以下所有维度均为 DeepSeek-V3 实际配置：$d_{\text{model}}=7168,\; n_h=128,\; d_h=128,\; d_c=512,\; d_c'=1536,\; d_r=64$

### A.0 预计算阶段（模型加载时，一次性完成）

将 KV 上采样矩阵吸收到 Q 投影和 O 投影中，生成吸收后的权重：

| 步骤 | 计算 | 输入维度 | 输出维度 | 说明 |
|------|------|----------|----------|------|
| P1 | $W_{Q,\text{abs}}^{(i)} = (W_{UK}^{(i)})^\top \cdot W_{UQ}^{(i)}$ | $[128 \times 512]^\top \cdot [128 \times 1536]$ | $[512 \times 1536]$ | 每头一个，共 128 个 |
| P2 | $W_{O,\text{abs}}^{(i)} = W_O^{(i)} \cdot W_{UV}^{(i)}$ | $[7168 \times 128] \cdot [128 \times 512]$ | $[7168 \times 512]$ | 每头一个，共 128 个 |

> 注：$W_{UK}^{(i)} \in \mathbb{R}^{128 \times 512}$ 是第 $i$ 头从 $d_c$ 到 $d_h$ 的上投影切片。

### A.1 Step 1：生成当前 token 的 KV 压缩表示并写入 Cache

| 操作 | 公式 | 权重矩阵维度 | 输入维度 | 输出维度 |
|------|------|-------------|----------|----------|
| KV 下投影 | $\mathbf{c}_t^{KV} = W_{DKV} \cdot \mathbf{h}_t$ | $W_{DKV} \in [512 \times 7168]$ | $\mathbf{h}_t \in [7168]$ | $\mathbf{c}_t^{KV} \in [512]$ |
| RoPE Key 投影 | $\mathbf{k}_t^{R,\text{pre}} = W_{KR} \cdot \mathbf{h}_t$ | $W_{KR} \in [64 \times 7168]$ | $\mathbf{h}_t \in [7168]$ | $\mathbf{k}_t^{R,\text{pre}} \in [64]$ |
| 应用 RoPE | $\mathbf{k}_t^R = \text{RoPE}(\mathbf{k}_t^{R,\text{pre}},\, \text{pos}_t)$ | — | $[64]$ | $\mathbf{k}_t^R \in [64]$ |

**写入 KV Cache**：追加 $(\mathbf{c}_t^{KV},\, \mathbf{k}_t^R)$，每 token 共 $512 + 64 = \mathbf{576}$ 个元素。

Cache 形状（到位置 $t$）：
- `c_KV_cache`: $[t \times 512]$
- `k_R_cache`: $[t \times 64]$

### A.2 Step 2：计算 Query（使用吸收后权重）

| 操作 | 公式 | 权重矩阵维度 | 输入维度 | 输出维度 |
|------|------|-------------|----------|----------|
| Q 下投影 | $\mathbf{c}_t^Q = W_{DQ} \cdot \mathbf{h}_t$ | $W_{DQ} \in [1536 \times 7168]$ | $\mathbf{h}_t \in [7168]$ | $\mathbf{c}_t^Q \in [1536]$ |
| 吸收后 Q（每头） | $\mathbf{q}_{t,\text{abs}}^{(i)} = W_{Q,\text{abs}}^{(i)} \cdot \mathbf{c}_t^Q$ | $W_{Q,\text{abs}}^{(i)} \in [512 \times 1536]$ | $\mathbf{c}_t^Q \in [1536]$ | $\mathbf{q}_{t,\text{abs}}^{(i)} \in [512]$ |
| RoPE Q 投影（每头） | $\mathbf{q}_t^{R,(i),\text{pre}} = W_{QR}^{(i)} \cdot \mathbf{c}_t^Q$ | $W_{QR}^{(i)} \in [64 \times 1536]$ | $\mathbf{c}_t^Q \in [1536]$ | $\mathbf{q}_t^{R,(i),\text{pre}} \in [64]$ |
| 应用 RoPE | $\mathbf{q}_t^{R,(i)} = \text{RoPE}(\mathbf{q}_t^{R,(i),\text{pre}},\, \text{pos}_t)$ | — | $[64]$ | $\mathbf{q}_t^{R,(i)} \in [64]$ |

所有 128 个头并行：
- 吸收后内容 Query: $Q_{\text{abs}} \in [128 \times 512]$
- RoPE Query: $Q_R \in [128 \times 64]$

### A.3 Step 3：计算注意力分数（在压缩空间直接计算）

对每个头 $i$，对所有缓存位置 $j \in \{1, \ldots, t\}$：

$$
\text{score}_{t,j}^{(i)} = \underbrace{(\mathbf{q}_{t,\text{abs}}^{(i)})^\top \cdot \mathbf{c}_j^{KV}}_{\text{内容分数: } [512]^\top \cdot [512] \to \text{scalar}} + \underbrace{(\mathbf{q}_t^{R,(i)})^\top \cdot \mathbf{k}_j^R}_{\text{位置分数: } [64]^\top \cdot [64] \to \text{scalar}}
$$

$$
\text{score}_{t,j}^{(i)} \mathrel{/}= \sqrt{d_h + d_r} = \sqrt{128 + 64} = \sqrt{192}
$$

矩阵化表示（所有头并行）：

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| 内容分数 | $S_{\text{content}} = Q_{\text{abs}} \cdot C_{\text{cache}}^\top$ | $[128 \times 512] \cdot [512 \times t] \to [128 \times t]$ |
| 位置分数 | $S_{\text{pos}} = Q_R \cdot K_{R,\text{cache}}^\top$ | $[128 \times 64] \cdot [64 \times t] \to [128 \times t]$ |
| 总分数 | $S = (S_{\text{content}} + S_{\text{pos}}) / \sqrt{192}$ | $[128 \times t]$ |

### A.4 Step 4：Softmax

$$
\alpha_{t,j}^{(i)} = \text{softmax}_j(S^{(i)}_{t,:}) \quad \in [128 \times t]
$$

### A.5 Step 5：在压缩空间聚合 Value（不解压！）

$$
\mathbf{o}_{\text{comp}}^{(i)} = \sum_{j=1}^{t} \alpha_{t,j}^{(i)} \cdot \mathbf{c}_j^{KV} \quad \in \mathbb{R}^{512}
$$

矩阵化：

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| 压缩空间聚合 | $O_{\text{comp}} = A \cdot C_{\text{cache}}$ | $[128 \times t] \cdot [t \times 512] \to [128 \times 512]$ |

**关键：此处直接用 $\mathbf{c}_j^{KV}$（512维压缩态）进行加权求和，而非解压后的 $\mathbf{v}_j$（128维/头 × 128头 = 16384维）。**

### A.6 Step 6：吸收后输出投影

$$
\mathbf{o}_t = \sum_{i=1}^{128} W_{O,\text{abs}}^{(i)} \cdot \mathbf{o}_{\text{comp}}^{(i)} \quad \in \mathbb{R}^{7168}
$$

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| 吸收后输出投影 | 每头: $W_{O,\text{abs}}^{(i)} \cdot \mathbf{o}_{\text{comp}}^{(i)}$ | $[7168 \times 512] \cdot [512] \to [7168]$ |
| 多头求和 | $\mathbf{o}_t = \sum_i \mathbf{o}_t^{(i)}$ | 128 个 $[7168] \to [7168]$ |

### A.7 完整数据流总览图

```
h_t [7168]
 ├─→ W_DKV [512×7168] ──→ c_t^KV [512] ──→ 写入 Cache
 ├─→ W_KR  [64×7168]  ──→ RoPE ──→ k_t^R [64] ──→ 写入 Cache
 └─→ W_DQ  [1536×7168] ──→ c_t^Q [1536]
      ├─→ W_Q_abs[i] [512×1536] ──→ q_abs[i] [512] ─┐
      └─→ W_QR[i] [64×1536] ──→ RoPE ──→ q_R[i] [64]─┤
                                                        ↓
              ┌─────────── Attention Score ←────────────┘
              │    q_abs[i]·c_j^KV + q_R[i]·k_j^R → scalar
              ↓
         Softmax → α [128 × t]
              │
              ↓
         α · C_cache [t×512] → o_comp[i] [512]
              │
              ↓
         W_O_abs[i] [7168×512] · o_comp[i] → o_t^(i) [7168]
              │
              ↓ (128头求和)
         o_t [7168] → 输出
```

---

## 附录 B：Qwen3-235B-A22B GQA Decode 逐步计算流程（标注全部维度）

### B.0 Qwen3-235B-A22B 注意力配置

| 参数 | 值 | 说明 |
|------|-----|------|
| $d_{\text{model}}$ | 4096 | 隐藏层维度 |
| $n_h$ | 64 | Query 头数 |
| $n_{\text{kv}}$ | 4 | KV 头数 |
| $d_h$ | 128 | 每头维度 ($d_{\text{model}} / n_h = 4096 / 64$... 实际 head_dim=128) |
| $n_{\text{layers}}$ | 94 | 层数 |
| GQA 比率 | 64 : 4 = 16 : 1 | 每个 KV 头被 16 个 Q 头共享 |

> 注：Qwen3-235B 的 head_dim=128 是独立配置的（$n_h \times d_h = 64 \times 128 = 8192 \neq d_{\text{model}} = 4096$），这意味着 Q/K/V 投影矩阵将 4096 维映射到 8192 维（Q）或 512 维（K/V），不等于 $d_{\text{model}}$。

### B.1 Step 1：投影 Q, K, V

| 操作 | 公式 | 权重矩阵维度 | 输入维度 | 输出维度 |
|------|------|-------------|----------|----------|
| Q 投影 | $Q = \mathbf{h}_t \cdot W_Q^\top$ | $W_Q \in [n_h \cdot d_h \times d_{\text{model}}] = [8192 \times 4096]$ | $\mathbf{h}_t \in [4096]$ | $Q \in [8192]$ → reshape $[64 \times 128]$ |
| K 投影 | $K_{\text{new}} = \mathbf{h}_t \cdot W_K^\top$ | $W_K \in [n_{\text{kv}} \cdot d_h \times d_{\text{model}}] = [512 \times 4096]$ | $\mathbf{h}_t \in [4096]$ | $K_{\text{new}} \in [512]$ → reshape $[4 \times 128]$ |
| V 投影 | $V_{\text{new}} = \mathbf{h}_t \cdot W_V^\top$ | $W_V \in [n_{\text{kv}} \cdot d_h \times d_{\text{model}}] = [512 \times 4096]$ | $\mathbf{h}_t \in [4096]$ | $V_{\text{new}} \in [512]$ → reshape $[4 \times 128]$ |

### B.2 Step 2：应用 RoPE

| 操作 | 输入维度 | 输出维度 |
|------|----------|----------|
| $Q = \text{RoPE}(Q, \text{pos}_t)$ | $[64 \times 128]$ | $[64 \times 128]$ |
| $K_{\text{new}} = \text{RoPE}(K_{\text{new}}, \text{pos}_t)$ | $[4 \times 128]$ | $[4 \times 128]$ |

### B.3 Step 3：写入 KV Cache

追加 $(K_{\text{new}}, V_{\text{new}})$ 到 Cache：

- `K_cache`: $[4 \times t \times 128]$（4 个 KV 头，每头 128 维）
- `V_cache`: $[4 \times t \times 128]$

**每 token 每层缓存**：$2 \times n_{\text{kv}} \times d_h = 2 \times 4 \times 128 = \mathbf{1024}$ 个元素

### B.4 Step 4：计算注意力分数（KV 头广播）

GQA 的核心：每个 KV 头被 $n_h / n_{\text{kv}} = 64 / 4 = 16$ 个 Q 头共享。

```
Q:       [64 × 128]     → 分为 4 组，每组 16 个 Q 头
K_cache: [4 × t × 128]  → 每组 1 个 KV 头广播到 16 个 Q 头
```

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| 分组 | $Q$ reshape 为 $[4 \times 16 \times 128]$ | Q 头按 KV 组分组 |
| 注意力分数 | $S_g = Q_g \cdot K_g^\top$ | $[16 \times 128] \cdot [128 \times t] \to [16 \times t]$，每组 |
| 缩放 | $S_g = S_g / \sqrt{d_h} = S_g / \sqrt{128}$ | $[16 \times t]$ |

全部 4 组并行：$S \in [4 \times 16 \times t] = [64 \times t]$

### B.5 Step 5：Softmax

$$
\alpha_{t,j}^{(i)} = \text{softmax}_j(S^{(i)}_{t,:}) \quad \in [64 \times t]
$$

### B.6 Step 6：加权聚合 Value

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| 每组聚合 | $O_g = A_g \cdot V_g$ | $[16 \times t] \cdot [t \times 128] \to [16 \times 128]$，每组 |
| 全部 4 组 | $O \in [4 \times 16 \times 128] = [64 \times 128]$ | |

### B.7 Step 7：输出投影

| 操作 | 计算 | 维度变化 |
|------|------|----------|
| Reshape | $O$ flatten 为 $[8192]$ | $[64 \times 128] \to [8192]$ |
| 输出投影 | $\mathbf{o}_t = O \cdot W_O^\top$ | $[8192] \cdot [8192 \times 4096]^\top \to [4096]$ |

> 注：$W_O \in [d_{\text{model}} \times n_h \cdot d_h] = [4096 \times 8192]$

### B.8 完整数据流总览图

```
h_t [4096]
 ├─→ W_Q [8192×4096] ──→ Q [64×128] ──→ RoPE ──→ Q [64×128]
 ├─→ W_K [512×4096]  ──→ K [4×128]  ──→ RoPE ──→ K [4×128] → 写入 Cache
 └─→ W_V [512×4096]  ──→ V [4×128]  ─────────────────────── → 写入 Cache

         Q [64×128]  ←→  K_cache [4×t×128]（每 KV 头广播到 16 个 Q 头）
              │
              ↓
         Score = Q · K^T / √128 → [64 × t]
              │
              ↓
         Softmax → α [64 × t]
              │
              ↓
         α · V_cache [4×t×128]（广播）→ O [64 × 128]
              │
              ↓
         W_O [4096×8192] · O.flatten [8192] → o_t [4096]
```

---

## 附录 C：DeepSeek-V3 MLA vs Qwen3-235B GQA 详细对比

### C.1 架构参数对比

| 参数 | DeepSeek-V3 (MLA) | Qwen3-235B-A22B (GQA) |
|------|-------------------|----------------------|
| 模型类型 | MoE (671B/37B active) | MoE (235B/22B active) |
| $d_{\text{model}}$ | 7168 | 4096 |
| Q 头数 $n_h$ | 128 | 64 |
| KV 头数 | N/A（无独立 KV 头） | 4 |
| 每头维度 $d_h$ | 128 | 128 |
| GQA 比率 | N/A | 16:1 |
| 层数 | 61 | 94 |
| 位置编码 | RoPE（解耦，$d_r=64$） | RoPE（直接应用，$d_h=128$） |

### C.2 KV Cache 对比（核心差异）

| 指标 | DeepSeek-V3 (MLA) | Qwen3-235B (GQA) |
|------|-------------------|------------------|
| **缓存内容** | 压缩潜变量 $\mathbf{c}^{KV}$ + RoPE key $\mathbf{k}^R$ | 完整 K 和 V 头 |
| **每 token 每层** | $d_c + d_r = 512 + 64 = \mathbf{576}$ 元素 | $2 \times n_{\text{kv}} \times d_h = 2 \times 4 \times 128 = \mathbf{1024}$ 元素 |
| **每 token 全模型** | $576 \times 61 = 35{,}136$ 元素 | $1024 \times 94 = 96{,}256$ 元素 |
| **BF16 下每 token** | $35{,}136 \times 2\text{B} = \mathbf{68.6\text{ KB}}$ | $96{,}256 \times 2\text{B} = \mathbf{188.0\text{ KB}}$ |
| **128K 上下文总量** | $\approx \mathbf{8.6\text{ GB}}$ | $\approx \mathbf{23.5\text{ GB}}$ |

> DeepSeek-V3 虽然模型更大（7168 vs 4096 hidden size, 128 vs 64 Q heads），但 KV Cache 反而只有 Qwen3-235B 的 **36.5%**。

### C.3 Decode 计算流程对比（简化版）

> **seq** = 已缓存序列长度。RoPE 不改变维度，已合并到 GEMM 中不单独标注。MLA 的下投影 + 吸收后上投影合并为单个等效矩阵。

#### ① Proj Q

| | DeepSeek-V3 MLA | Qwen3-235B GQA |
|---|---|---|
| 输入 | $\mathbf{h} \in [7168]$ | $\mathbf{h} \in [4096]$ |
| 等效权重 | $W_Q \in [73728 \times 7168]$ | $W_Q \in [8192 \times 4096]$ |
| 输出 | $Q \in [73728]$，reshape $[128 \times 576]$ | $Q \in [8192]$，reshape $[64 \times 128]$ |
| 每头 Q 维度 | 576 = 512(内容,已吸收$W_{UK}$) + 64(RoPE) | 128（全部含 RoPE） |

> MLA: $128 \times 576 = 73728$；GQA: $64 \times 128 = 8192$

#### ② Proj KV → 写入 Cache

| | DeepSeek-V3 MLA | Qwen3-235B GQA |
|---|---|---|
| 输入 | $\mathbf{h} \in [7168]$ | $\mathbf{h} \in [4096]$ |
| 等效权重 | $W_{KV} \in [576 \times 7168]$ | $W_K [512 \times 4096] + W_V [512 \times 4096]$，合计 $[1024 \times 4096]$ |
| 输出 / 写入 Cache | $[576]$ = $\mathbf{c}^{KV}[512]$(K/V联合) + $\mathbf{k}^R[64]$(RoPE) | $[1024]$ = $K[4 \times 128] + V[4 \times 128]$ |
| Cache 形状 | $[\text{seq} \times 576]$ | $K[\text{seq} \times 512] + V[\text{seq} \times 512]$ |
| **每 token 缓存量** | **576** | **1024** |

#### ③ Core Attn

| | DeepSeek-V3 MLA | Qwen3-235B GQA |
|---|---|---|
| **Score** | $Q[128 \times 576] \cdot \text{Cache}[\text{seq} \times 576]^\top$ | $Q[64 \times 128] \cdot K[\text{seq} \times 128]^\top$（KV头广播） |
| | $\to S \in [128 \times \text{seq}]$ | $\to S \in [64 \times \text{seq}]$ |
| **Softmax** | $\alpha \in [128 \times \text{seq}]$ | $\alpha \in [64 \times \text{seq}]$ |
| **Value 聚合** | $\alpha[128 \times \text{seq}] \cdot C_{\text{cache}}[\text{seq} \times 512]$ | $\alpha[64 \times \text{seq}] \cdot V[\text{seq} \times 128]$（KV头广播） |
| | $\to O \in [128 \times 512]$ | $\to O \in [64 \times 128]$ |

> MLA 注意：Score 用完整 576 维 Cache（内容512 + RoPE64），但 Value 聚合只用前 512 维（$\mathbf{c}^{KV}$ 部分），RoPE 分量仅编码位置、不携带 Value 信息。

#### ④ Proj O

| | DeepSeek-V3 MLA | Qwen3-235B GQA |
|---|---|---|
| 输入 | $O$ flatten $\to [65536]$（$128 \times 512$） | $O$ flatten $\to [8192]$（$64 \times 128$） |
| 等效权重 | $W_O \in [7168 \times 65536]$（已吸收 $W_{UV}$） | $W_O \in [4096 \times 8192]$ |
| 输出 | $\mathbf{o} \in [7168]$ | $\mathbf{o} \in [4096]$ |

### C.4 计算量与访存对比分析

#### 注意力分数计算（per token, per layer）

| 模型 | 操作 | FLOPs |
|------|------|-------|
| DeepSeek-V3 | 内容: $128 \times 2 \times 512 \times t = 131{,}072 \cdot t$ | $\mathbf{147{,}456 \cdot t}$ |
| | 位置: $128 \times 2 \times 64 \times t = 16{,}384 \cdot t$ | |
| Qwen3-235B | $64 \times 2 \times 128 \times t = 16{,}384 \cdot t$ | $\mathbf{16{,}384 \cdot t}$ |

> MLA 的注意力分数计算量约为 GQA 的 **9 倍**，因为 MLA 在 512 维压缩空间做点积（每个头独立），而 GQA 在 128 维做点积（KV 头广播共享）。这是 MLA 用计算换显存的 tradeoff。

#### Value 聚合（per token, per layer）

| 模型 | 操作 | FLOPs |
|------|------|-------|
| DeepSeek-V3 | $128 \times 2 \times 512 \times t = 131{,}072 \cdot t$ | $\mathbf{131{,}072 \cdot t}$ |
| Qwen3-235B | $64 \times 2 \times 128 \times t = 16{,}384 \cdot t$ | $\mathbf{16{,}384 \cdot t}$ |

#### KV Cache 访存量（per token, per layer, BF16）

| 模型 | 每步需读取的 Cache | 数据量 |
|------|-------------------|--------|
| DeepSeek-V3 | $t \times 576 \times 2\text{B}$ | $\mathbf{1{,}152 \cdot t}$ B |
| Qwen3-235B | $t \times 1024 \times 2\text{B}$ | $\mathbf{2{,}048 \cdot t}$ B |

> MLA 虽然计算量更大，但 KV Cache 访存量只有 GQA 的 **56%**。在 decode 阶段（memory-bound），更少的访存往往比更少的计算更重要。

### C.5 设计哲学差异总结

| 维度 | DeepSeek-V3 MLA | Qwen3-235B GQA |
|------|----------------|----------------|
| **核心思路** | 低秩压缩 + 矩阵吸收 | 头共享（简单有效） |
| **KV Cache 压缩** | 学习到的联合压缩，信息保留多 | 减少 KV 头数，信息有损 |
| **每头信息独立性** | 每头通过独立 $W_{UK}^{(i)}, W_{UV}^{(i)}$ 从共享潜变量提取**不同**的 K/V | 同组 Q 头看到**完全相同**的 K/V |
| **实现复杂度** | 高（需预计算吸收矩阵、解耦 RoPE） | 低（标准注意力 + 头广播） |
| **推理优化** | 需要专门的 kernel 支持压缩空间注意力 | 标准 FlashAttention 等可直接使用 |
| **适合场景** | 极长序列、大 batch 推理（KV Cache 是瓶颈） | 通用场景（实现简单，生态成熟） |
| **性价比** | 用额外计算换取显存节省 | 用少量质量损失换取显存节省 |
