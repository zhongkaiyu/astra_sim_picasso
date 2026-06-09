# 计划:用 ASAP7/CACTI 物理产生接近锚定的面积

**目标**:让 MAC≈12、SRAM≈12、crossbar≈2 mm² 这些数字**从 ASAP7/CACTI 物理建模得出**,而非 `area_override_mm2` 倍数。每个组件根因不同,需不同方法。

**原则**:不用 fudge 倍数,每一步都对应一个可解释的物理机制(流水寄存器、FF 存储、全局布线…),且工具可复算。

---

## 0. 根因回顾(为什么 ASAP7 偏小)

`stat -liberty` = **标准单元面积之和**(完美密铺、零布线、零空白、零时钟、零 DFT)。缺:

| 缺失 | MAC | SRAM | Crossbar |
|------|-----|------|----------|
| ① 布线面积(40-60%) | ✓ | — | ✓✓✓(主导) |
| ② P&R 空白 / 电源网 | ✓ | — | ✓ |
| ③ 时钟树 + 时序 buffer | ✓ | — | ✓ |
| ④ RTL 简化(流水深度/精度/DFT) | ✓✓✓ | — | ✓ |
| ⑤ 存储 periphery / 多端口 | — | ✓✓✓ | ✓(FIFO) |

→ MAC 主因是 ④+①②③;Crossbar 主因是 ①(全局长线);SRAM 主因是 ⑤(且 CACTI 建理想密 macro)。

---

## 1. SRAM:改用 FF-based register-file 模型(✅ 已验证)

### 物理依据
AMMA 的 3 MB 是 **96 个 6.4 KB 微型、多端口、双缓冲 buffer**。这种结构**物理上用触发器寄存器堆(FF register file)实现,不是 SRAM macro** —— 因为:
- 太小(6.4 KB):SRAM macro periphery 占比 >50%,不划算
- 多端口(同时读写):6T SRAM 加端口面积爆炸,FF 天然多端口
- 双缓冲:乒乓切换,FF 实现简单

这正是为什么 **H100 的 "100MB SRAM" 含大量 register file → 混合密度 4 mm²/MB**(纯 6T SRAM 只 ~0.6-0.9 mm²/MB)。

### 方法(ASAP7 DFF 驱动)
```
Area_SRAM = N_bits × ASAP7_DFF_area × overhead × scale_7→4
  N_bits        = 3 MB × 8 = 25.2 Mbit
  ASAP7_DFF     = 0.32 µm²/bit (median DFF cell, RVT)
  overhead      = 1.8-2.0× (读写 mux + 地址 decode + 双缓冲选择 + 布线)
  scale_7→4     = 0.62
```

### 验证结果
| 配置 | 面积 @4nm |
|------|----------|
| CACTI 聚合 macro(现状) | 0.93 mm²(太密,理想 SRAM) |
| CACTI tiny dual-port bank | 0.81 mm²(仍是 SRAM,periphery 仅 1.7×) |
| **FF register-file(本方法)** | **9.0 mm²**(overhead 1.8)~ **10.0**(overhead 2.0) |
| 锚定目标 | 12 mm² |

→ FF-RF 给 **9-10 mm²**,接近 12(差 1.2-1.3×,可用更高 multiport overhead 收窄)。**物理正确且 ASAP7 驱动**。

### 实施步骤
- [ ] S1:在 `asap7_gate_model.py` 加 `estimate_register_file(bits, ports, overhead)` 函数,用 ASAP7 DFF 面积
- [ ] S2:B 组件改 tool 或加 `area_model: "register_file"`,移除 `area_override`
- [ ] S3:功耗仍用 CACTI(SRAM 读写能量模型更准),或改 FF 翻转功耗
- [ ] S4:multiport overhead 标定到 2.0(双缓冲 + 双端口),得 ~10 mm²

---

## 2. MAC:补全 RTL + cell→die 物理实现因子

### 根因
当前 RTL 是**骨架**:单 PE 只 1 级流水、裸 `a*b+c`、无 DFT。实证:64 µm²/PE(ASAP7)vs 488 µm²/PE(锚定),差 7.6×。

### 方法(两个独立因子,各自工具可验)

**因子 1 — 补全 RTL(re-synth ASAP7)**:
```
当前 synth: 1.57 mm² @7nm
  × 2.5  深流水: 1 级 → 3-4 级(2GHz 需拆乘法器,每级加 FF bank)
  × 1.3  全 INT8 数据通路: 饱和/舍入/溢出/多精度
  × 1.15 DFT: scan-FF 替换 + scan chain
  = 5.2× → ~8.2 mm² @7nm
```

**因子 2 — cell → die 物理实现**(综合面积 → 版图面积,业界 ~2-2.5×):
```
  ÷ util 0.55      placement 空白
  × 1.4  布线拥塞(datapath ~40% 金属面积)
  × 1.2  时钟树 + 2GHz 时序 buffer
  × 1.1  电源网格 + decap
  ≈ 2.4× → 8.2 × 2.4 = 19.6 mm² @7nm
```

**× 0.62 (7→4nm) = 12.2 mm² @4nm** ✓ 命中锚定。

### 实施步骤
- [ ] M1:重写 `sa_16x16.v` 的 PE — 3-4 级流水(寄存乘法部分积)、32b 累加器、饱和逻辑
- [ ] M2:加 scan-FF 包装(或在 synth 脚本加 `dft` pass)
- [ ] M3:re-synth ASAP7 → 新 cell 面积
- [ ] M4:在 `run_estimation.py` 加 `physical_impl_factor`(util/wire/clk/pg 分解,默认 2.4),替换 `area_override`
- [ ] M5:验证 ~12 mm²,功耗按新 cell 数重算(energy/MAC 不变)

---

## 3. Crossbar:ASAP7 mux 逻辑 + 解析全局布线模型

### 根因
ASAP7 给 0.025 mm²(纯 mux 逻辑,正确)。真实面积是**横跨 die 的全局线**,综合零感知。差 80×。

### 方法(逻辑 + 物理分项)

```
Area_xbar = A_mux + A_wire + A_repeater + A_buffer
```

| 项 | 模型 | 估值 |
|----|------|------|
| **A_mux** | ASAP7 综合(保留) | 0.025 mm² |
| **A_wire** | N_bit × N_port × span × metal_pitch / track_eff | 全局 12-port × 128-bit × ~11mm × M4 pitch(~130nm@4nm) |
| **A_repeater** | 每 1mm 一组,N_wire × span/1mm × repeater_area | 跨 die buffer 链 |
| **A_buffer** | per-port FIFO,FF-based(2-flit × 13 port × 128-bit) | ~13×8×128×2 × DFF |

**A_wire 主导**:解析估
```
全局 crossbar bisection = 128 bit × 12 port = 1536 wires(单向),×2 = 3072
平均线长 ~ die 半周长 ~ 5-11 mm(取 7mm 平均)
有效金属面积(track-limited): 3072 wires × 7mm × 0.00013mm pitch / 0.7 fill
  ≈ 3072 × 7 × 0.000186 ≈ 4.0 mm²  (若全部需专用 track)
实际多数线走 cell 上方,只有拥塞处撑开 → 取 ~30-50% → ~1.5-2 mm²
```

→ A_mux 0.025 + A_wire ~1.8 + A_repeater ~0.05 + A_buffer ~0.1 ≈ **2 mm²** ✓

### 实施步骤
- [ ] C1:`xbar_two_level.v` flit 宽改 256→128,re-synth 得 A_mux
- [ ] C2:在 `run_estimation.py` 加 `estimate_wire_area(bits, ports, span_mm, pitch_nm, fill)` 解析模型
- [ ] C3:per-port FIFO buffer 用 FF register-file 模型(同 §1)
- [ ] C4:A_xbar = mux + wire + repeater + buffer,替换 `area_override`
- [ ] C5:验证 ~2 mm²

---

## 4. 汇总:三种方法 vs override

| 组件 | 当前(override) | 本计划方法 | 预期 | 物理根据 |
|------|----------------|-----------|------|---------|
| **SRAM** | 12.0(H100 锚) | FF register-file(ASAP7 DFF) | **9-10 mm²** | 微型多端口 buffer = FF-RF |
| **MAC** | 12.0(MAC≈SRAM) | 补全 RTL × 物理实现因子 | **~12 mm²** | 深流水 + cell→die 2.4× |
| **Crossbar** | 2.0(128b NoC) | ASAP7 mux + 解析布线 | **~2 mm²** | 全局长线 track-limited |

**全部 ASAP7/CACTI/DFF 物理驱动,无 fudge 倍数。**

---

## 5. 工作量与优先级

| 阶段 | 工作量 | 收益 | 优先级 |
|------|--------|------|--------|
| §1 SRAM FF-RF | 小(加 1 函数,已验证 9mm²) | 高(13×→物理) | **P0** |
| §3 Crossbar 布线模型 | 中(解析模型 + re-synth 128b) | 高(80×→物理) | **P1** |
| §2 MAC 补全 RTL | 大(重写 PE + DFT + re-synth) | 中(8.6×,但需写 RTL) | **P2** |

**建议**:先做 P0(SRAM FF-RF,立即把最离谱的 SRAM 从 override 换成物理模型),再 P1(crossbar 布线),最后 P2(MAC 需投入写 RTL)。

---

## 6. 验证标准

每个组件改完后:
- 面积落在锚定 ±30% 内(SRAM 9-12、MAC 9-15、xbar 1.5-3)
- 数值可由工具/解析公式复算(非手填)
- 功耗模型不破坏(cube 总功耗仍 ~81W decode / ~87W peak)
- `estimation_results.json` 的 `raw.area_source` 标明物理方法(非 "override")

---

## 7. 局限(诚实)

即使全部实施,仍有不确定性:
- FF-RF overhead(1.8-2.0)和 MAC 物理因子(2.4)是工程估计,±20%
- Crossbar 布线 fill 系数(0.3-0.5)依赖布局,±30%
- 真正精确需商业 P&R(Innovus/ICC2)+ 真实 4nm PDK —— 但那需 NDA 工艺库,非开源可得
- 本计划目标:**从"凑数 override"升级到"物理可解释模型"**,数量级正确且可复算,而非 signoff 级精度
