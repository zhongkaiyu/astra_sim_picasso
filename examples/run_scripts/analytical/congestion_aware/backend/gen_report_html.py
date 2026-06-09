#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_report_html.py — 生成 backend/experiments_report.html。

一页式报告，描述 fig1 / exp1 / exp3 / exp4(new baselines) 的脚本依赖与参数 config，
并把 exp4 baseline 结果、exp1、fig1、exp3 的图（已有图片直接 base64 内嵌）+ 原始数据表格贴上。

运行: python3 backend/gen_report_html.py
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))          # congestion_aware/
FIG = os.path.join(ROOT, "figures")


def img64(relpath):
    """读图 → base64 <img> data-uri；找不到则返回提示。"""
    p = os.path.join(ROOT, relpath)
    if not os.path.exists(p):
        return f'<p style="color:red">[missing image: {relpath}]</p>'
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{os.path.basename(relpath)}"/>'


def load_json(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return json.load(f)


# ===========================================================================
# 数据表（从 JSON 现算）
# ===========================================================================
SEQS = [2048, 8192, 32768, 131072, 1048576]
SEQL = ["2K", "8K", "32K", "128K", "1M"]
MODELS = ["deepseek3", "qwen3-235b"]
MTITLE = {"deepseek3": "DeepSeek-V3 (MLA)", "qwen3-235b": "Qwen3-235B (GQA)"}
ORDER = [("gpu_gpu", "GPU+GPU (exp1)"), ("gpu_lpu", "GPU+LPU (exp1)"),
         ("duplex", "Duplex (exp4)"), ("helios", "Helios (exp4)"),
         ("stratum", "Stratum (exp4)"), ("amma_lpu", "AMMA+LPU ours (exp1)")]


def tpot_table():
    e1 = load_json("figures/rebuttal_figures/exp1_e2e_decode/data/baseline_three_settings.json")
    e4 = load_json("figures/rebuttal_figures/exp4_newbaseline/data/nmp_baselines.json")
    rows = []
    for m in MODELS:
        for s, label in ORDER:
            if s in ("gpu_gpu", "gpu_lpu", "amma_lpu"):
                vals = [e1["data"][m][str(cl)][s]["tpot_us"] for cl in SEQS]
            else:
                rm = {r["seq"]: r["tpot_us"] for r in e4["data"][s][m]["rows"]}
                vals = [rm[cl] for cl in SEQS]
            tds = "".join(f"<td>{v:,.0f}</td>" for v in vals)
            cls = ' class="ours"' if s == "amma_lpu" else (' class="new"' if s in ("duplex", "helios", "stratum") else "")
            rows.append(f'<tr{cls}><td>{MTITLE[m]}</td><td>{label}</td>{tds}</tr>')
    head = "".join(f"<th>CL={x}</th>" for x in SEQL)
    return ('<table><thead><tr><th>Model</th><th>Setting</th>' + head +
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')


def breakdown_table():
    e4 = load_json("figures/rebuttal_figures/exp4_newbaseline/data/nmp_baselines.json")
    comps = ["proj_qkv", "attention", "proj_o", "attn_noc", "ffn", "ffn_noc"]
    rows = []
    for b in ["duplex", "helios", "stratum"]:
        for m in MODELS:
            for r in e4["data"][b][m]["rows"]:
                if r["seq"] not in (8192, 131072, 1048576):
                    continue
                bd = r["breakdown_us"]
                tds = "".join(f"<td>{bd[c]:,.0f}</td>" for c in comps)
                rows.append(f'<tr><td>{e4["data"][b][m]["label"]}</td><td>{MTITLE[m]}</td>'
                            f'<td>{r["seq"]//1024}K</td>{tds}<td><b>{r["tpot_us"]:,.0f}</b></td></tr>')
    head = "".join(f"<th>{c}</th>" for c in comps)
    return ('<table><thead><tr><th>Baseline</th><th>Model</th><th>CL</th>' + head +
            '<th>TPOT(µs)</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>')


def fig1_mla_table():
    d = load_json("figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json")
    strat = d["strategies"]
    keys = ["h100", "h100_tp2", "rubin", "rubin_tp2", "hmp_reo_new"]
    lab = {"h100": "H100", "h100_tp2": "H100 TP2", "rubin": "Rubin",
           "rubin_tp2": "Rubin TP2", "hmp_reo_new": "Ours (16-NPU)"}
    seqs = [1024, 8192, 65536, 262144]
    rows = []
    for k in keys:
        db = strat[k].get("data_by_bs", {}).get("bs1", [])
        sm = {e["seq"]: e["hybrid_wall_ns"] / 1e3 for e in db}
        tds = "".join(f"<td>{sm.get(s, float('nan')):,.1f}</td>" for s in seqs)
        cls = ' class="ours"' if k == "hmp_reo_new" else ""
        rows.append(f'<tr{cls}><td>{lab[k]}</td>{tds}</tr>')
    head = "".join(f"<th>seq={s}</th>" for s in seqs)
    return ('<p>DeepSeek-V3 单层 decode attention wall (µs), batch=1（fig1 权威数据 '
            '<code>deepseek_v3_mla_fig1.json</code>）：</p>'
            '<table><thead><tr><th>Strategy</th>' + head + '</tr></thead><tbody>'
            + "".join(rows) + '</tbody></table>')


# ===========================================================================
# HTML 组装
# ===========================================================================
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1180px;
 margin:0 auto;padding:24px 32px;color:#1a1a1a;line-height:1.55;}
h1{border-bottom:3px solid #d62728;padding-bottom:8px;}
h2{margin-top:42px;border-bottom:2px solid #ccc;padding-bottom:5px;color:#b3271e;}
h3{margin-top:26px;color:#333;}
code{background:#f3f3f3;padding:1px 5px;border-radius:4px;font-size:90%;}
pre{background:#f7f7f9;border:1px solid #e1e1e8;border-radius:6px;padding:12px 14px;overflow:auto;font-size:12.5px;}
table{border-collapse:collapse;margin:14px 0;font-size:12.5px;width:100%;}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:right;}
th{background:#f0f0f4;text-align:center;}
td:first-child,td:nth-child(2){text-align:left;}
tr.ours{background:#fdeaea;font-weight:600;}
tr.new{background:#eef6ff;}
img{max-width:100%;border:1px solid #e0e0e0;border-radius:6px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.nav{background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;padding:10px 18px;margin:18px 0;}
.nav a{margin-right:16px;text-decoration:none;color:#b3271e;font-weight:600;}
.tag{display:inline-block;background:#607d8b;color:#fff;border-radius:4px;padding:1px 8px;font-size:11px;margin-left:6px;}
.tag.new{background:#1f77b4;} .tag.ours{background:#d62728;}
.note{background:#fff8e1;border-left:4px solid #ffb300;padding:8px 14px;margin:12px 0;font-size:13px;}
.dep{font-size:12.5px;}
"""


def section_overview():
    return f"""
<h1>Decode 加速器实验报告：fig1 / exp1 / exp3 + 新增 3 条 baseline (exp4)</h1>
<p>本页汇总四组实验的<b>脚本依赖、参数 config</b>，并贴上结果图（已有图片直接内嵌）与原始数据表格。
新增的 3 条 baseline（<b>Duplex / Helios / Stratum</b>）实现于 <code>backend/baselines/</code>，
对齐 fig1 的 attention roofline 与 exp1 的 e2e TPOT 组合框架。</p>
<div class="nav">
  <a href="#overview">统一框架</a><a href="#fig1">fig1</a><a href="#exp1">exp1</a>
  <a href="#exp3">exp3</a><a href="#exp4">exp4 (new)</a><a href="#caveats">caveats</a>
</div>

<h2 id="overview">0. 统一 Roofline 框架</h2>
<p>所有实验共享同一 roofline：每个 stage <code>t = max(FLOPs/peak, bytes/BW)</code>，
decode 小 GEMV 几乎总落 mem 边。GPU 侧按实测利用率 derate（<code>t/util</code>），
近存/SRAM 侧用 100% roofline（乐观上界）。整模 <code>TPOT = num_layers × t_layer</code>。</p>
<pre>fig1 (单层 attention speedup):  wall = QKV + Attn + ProjO + Comm,  stage = max(comp,mem)/util
exp1 (e2e decode TPOT):         t_layer = t_attn + xfer + t_ffn + xfer,  TPOT = L × t_layer
exp4 (NMP baselines):           t_layer = t_attn + t_ffn   (无跨池 NIC, attn+FFN 同址)  </pre>
"""


def section_fig1():
    cfg = """<table><thead><tr><th>方案</th><th>peak TFLOPS</th><th>HBM BW</th><th>dtype</th><th>互连</th></tr></thead>
<tbody>
<tr class="ours"><td>Ours (per NPU)</td><td>96</td><td>2.5 TB/s</td><td>FP8 (1B)</td><td>D2D UCIe 1.5 TB/s</td></tr>
<tr><td>H100 SXM5</td><td>1979</td><td>3.35 TB/s</td><td>BF16 (2B)</td><td>—</td></tr>
<tr><td>H100 TP2</td><td>1979×2</td><td>3.35×2</td><td>BF16</td><td>NVLink 0.9 TB/s, 800ns</td></tr>
<tr><td>Rubin</td><td>17500</td><td>22.0 TB/s</td><td>BF16</td><td>—</td></tr>
<tr><td>Rubin TP2</td><td>17500×2</td><td>22.0×2</td><td>BF16</td><td>NVLink 1.8 TB/s, 800ns</td></tr>
</tbody></table>"""
    return f"""
<h2 id="fig1">1. fig1 — 单层 decode attention 加速比 <span class="tag">paper_figures/fig1_e2e_latency</span></h2>
<p>单层 decode attention 的统一 roofline + 相对 H100 TP1 的 speedup，4-panel（Qwen3-235B / Llama4-Maverick /
DeepSeek-V3），策略 H100 / H100-TP2 / Rubin / Rubin-TP2 / Ours(16-NPU) / NeuPims。</p>
<h3>脚本依赖</h3>
<pre class="dep">backend/roofline/deepseek_v3_mla_roofline.py   # MLA absorbed roofline -> data/deepseek_v3_mla_fig1.json
                                               #   mla_single_gpu_baseline() / mla_roofline()
backend/roofline/attention.py                  # GQA 单层 roofline (single_layer_qkv/attention/output_time)
backend/hybrid_merge/merge_gqa_results.py      # GQA: AstraSim + roofline + util 合并 (需 AstraSim)
figures/paper_figures/fig1_e2e_latency/scripts/plot_e2e.py        # 主图
                                              /plot_pipeline.py, plot_e2e_higher_compute.py
util: roofline/utilization_profiles/{{qwen3,llama4,deepseek3}}/util_96.json   # Ours SA util
      roofline/utilization_profiles/h100/h100_rubin_utilization.json          # GQA BW util
      roofline/utilization_profiles/h100/h100_mla_profile.json                # MLA BW util</pre>
<h3>硬件参数 config</h3>
{cfg}
<h3>结果图（已有图片）</h3>
{img64("figures/paper_figures/fig1_e2e_latency/plots/fig1_speedup_4panel.png")}
{img64("figures/paper_figures/fig1_e2e_latency/plots/fig1_e2e_latency.png")}
<h3>原始数据表格</h3>
{fig1_mla_table()}
"""


def section_exp1():
    cfg = """<table><thead><tr><th>chip</th><th>角色</th><th>peak</th><th>mem BW</th><th>容量</th><th>dtype</th></tr></thead>
<tbody>
<tr><td>Rubin GPU</td><td>attn/FFN</td><td>17500 TFLOPS</td><td>HBM 22 TB/s</td><td>288 GB</td><td>FP8</td></tr>
<tr class="ours"><td>AMMA NPU (×16)</td><td>attn</td><td>96 TFLOPS/NPU</td><td>2.5 TB/s/NPU (40 agg)</td><td>36 GB</td><td>FP8</td></tr>
<tr><td>LPU LPX (×16)</td><td>FFN</td><td>1200 TFLOPS</td><td>SRAM 150 TB/s</td><td>0.5 GB</td><td>FP8</td></tr>
<tr><td>NIC cx7</td><td>跨池 link</td><td colspan="2">50 GB/s, 2000 ns 固定延迟</td><td colspan="2">decode payload ~KB → 延迟主导</td></tr>
</tbody></table>"""
    return f"""
<h2 id="exp1">2. exp1 — e2e decode TPOT（3 settings）<span class="tag">rebuttal_figures/exp1_e2e_decode</span></h2>
<p>整模 decode TPOT，三条 attention-pool↔FFN-pool 解耦 setting：GPU+GPU / GPU+LPU / AMMA+LPU(ours)，
池间统一走 NIC。<code>t_layer = t_attn + xfer + t_ffn + xfer</code>。</p>
<h3>脚本依赖</h3>
<pre class="dep">backend/lpu_ffn_decode/lpu_config.py       # LPUSpec / FFNModelSpec / RunSpec + 模型库
                      /ffn_decode.py        # FFN roofline (dense+MoE, OI=1, 100% roofline)  [LPU]
                      /rubin_ffn_decode.py  # FFN-on-GPU + 实卡 util derate                  [Rubin]
                      /decode_compose.py    # ours = AMMA attn + LPU FFN
                      /baseline_compose.py  # GPU+GPU / GPU+LPU
                      /interconnect.py      # 跨池 NIC/UCIe 传输模型
roofline/attention.py, roofline_gqa_calc.py # attention 单层 (GQA / MLA)
figures/rebuttal_figures/exp1_e2e_decode/scripts/gen_baseline_data.py -> data/baseline_three_settings.json
                                               /gen_decode_data.py, gen_result_md.py
                                               /plot_baseline_breakdown.py, plot_decode_breakdown.py
util: lpu_ffn_decode/data/gpu_ffn_utilization.json   # GPU FFN derate; LPU FFN = 100% roofline</pre>
<h3>硬件参数 config</h3>
{cfg}
<h3>结果图（已有图片）</h3>
{img64("figures/rebuttal_figures/exp1_e2e_decode/plots/baseline_three_settings_breakdown.png")}
{img64("figures/rebuttal_figures/exp1_e2e_decode/plots/decode_e2e_breakdown.png")}
{img64("figures/rebuttal_figures/exp1_e2e_decode/plots/decode_link_sensitivity.png")}
"""


def section_exp3():
    return f"""
<h2 id="exp3">3. exp3 — DeepSeek-V3 MLA 单层 decode 时间分解 <span class="tag">rebuttal_figures/exp3_MLA_breakdown</span></h2>
<p>MLA 单层 decode 时间分解，2×2（batch×seq），每柱 Proj QKV / Attention / Proj O / Comm，
5 策略 H100 / H100-TP2 / Rubin / Rubin-TP2 / Ours。</p>
<h3>脚本依赖</h3>
<pre class="dep">figures/rebuttal_figures/exp3_MLA_breakdown/plot_mla_breakdown.py
  └─ 数据复用 fig1: paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json
                    (strategies.&lt;key&gt;.data_by_bs.bs&lt;N&gt;)
arithmetic_intensity.md   # V4 CSA/HCA decode 计算强度 (AI=FLOPs/Bytes) 推导</pre>
<h3>结果图（已有图片）</h3>
{img64("figures/rebuttal_figures/exp3_MLA_breakdown/plots/exp3_mla_breakdown.png")}
"""


def section_exp4():
    cfg = """<table><thead><tr><th>baseline</th><th>内存技术</th><th>mem BW/设备</th><th>算力/设备</th>
<th>容量/设备</th><th>设备数 N (qwen3 / ds3)</th><th>NoC</th></tr></thead>
<tbody>
<tr class="new"><td>Duplex</td><td>HBM3 + 4×TSV (Logic-PIM)</td><td>13.4 TB/s</td><td>107 TFLOPS</td><td>80 GB</td><td>8 / 18</td><td>bank-bundle</td></tr>
<tr class="new"><td>Helios</td><td>Hybrid-Bonding 4-die 3D-DRAM</td><td>16.4 TB/s</td><td>~500 (占位)</td><td>80 GB</td><td>8 / 18</td><td>4×4 mesh</td></tr>
<tr class="new"><td>Stratum</td><td>Monolithic 3D DRAM (tiered)</td><td>28.4 TB/s*</td><td>~500 (占位)</td><td>32 GB/chip</td><td>15 / 44</td><td>双向 ring</td></tr>
</tbody></table>
<p style="font-size:12px">设备数 N = max(paper 规则, 容量可行)，逐模型（权重 FP16: qwen3-235B≈470GB, deepseek3≈1342GB）。
聚合带宽 = per-device BW × N：Duplex 107/241 · Helios 131/295 · <b>Stratum 425/1248 TB/s</b>（小芯片→多颗→带宽最高）。
* Stratum FFN 带宽 = tiering 命中率加权（p_hot=0.9 → 28.4 TB/s）；attention 用 25 TB/s（不享受 tiering）。全 FP16。</p>"""
    return f"""
<h2 id="exp4">4. exp4 — 新增 3 条 NMP baseline（本次实现）<span class="tag new">backend/baselines</span></h2>
<p><b>范式</b>：三者 decode 时 attention 与 FFN <b>同址跑在一块近存设备上</b>，无跨池 NIC →
<code>t_layer = t_attn + t_ffn</code>。这是与 exp1（每层走 NIC，~390µs）的核心结构差异。</p>
<h3>脚本依赖 + 调用链</h3>
<pre class="dep">backend/baselines/nmp_config.py     # Duplex/Helios/Stratum 的 NMPSpec(FP16) + attn_hw + TP/NoC + Stratum tiering
                  /nmp_compose.py    # 组合器: t_layer = t_attn_nmp + t_ffn_nmp
figures/rebuttal_figures/exp4_newbaseline/scripts/gen_nmp_baseline_data.py -> data/nmp_baselines.json
                                                 /plot_compare.py            -> plots/exp4_*.png
复用（对齐 fig1 + exp1）:
  attn GQA : roofline/attention.py (fused) + h100_rubin_utilization        [fig1 路径]
  attn MLA : roofline/deepseek_v3_mla_roofline.py:mla_single_gpu_baseline  [fig1 路径, 非 calc_mla_strategy]
             absorbed 权重 + h100_mla_profile，按 device 数(TP)除 + NoC AllReduce
  FFN      : lpu_ffn_decode/ffn_decode.py:simulate_layer (100% roofline)   [exp1 路径]
  NoC      : ffn_decode._allreduce_c2c (mesh/ring/bundle 参数)</pre>
<h3>硬件参数 config</h3>
{cfg}
<h3>结果图（新生成：exp4 三条 + exp1 三条合并对比）</h3>
{img64("figures/rebuttal_figures/exp4_newbaseline/plots/exp4_tpot_compare.png")}
{img64("figures/rebuttal_figures/exp4_newbaseline/plots/exp4_breakdown.png")}
<h3>原始数据表格 — Decode TPOT (µs)，6 setting 合并</h3>
{tpot_table()}
<h3>原始数据表格 — exp4 三条 baseline 延时分量 (µs，整模 e2e)</h3>
{breakdown_table()}
"""


def section_caveats():
    return """
<h2 id="caveats">5. Caveats（务必标注）</h2>
<ul>
<li><b>FFN util = 100% roofline（乐观上界）</b>：exp4 三条 NMP 与 exp1 LPU 同口径；GPU FFN 用实测 util derate。口径不对称，已在 meta 标 <code>ffn_util=100%_roofline_optimistic</code>。</li>
<li><b>attention util 不对称</b>：attention 被 H100 实测 util derate（GQA→h100_rubin_utilization，MLA→h100_mla_profile），继承自 exp1/fig1，三条 baseline 沿用 → 公平。</li>
<li><b>Helios/Stratum 算力为占位大值</b>：paper 未给干净 FP16 TFLOPS，decode mem-bound 故对算力不敏感。</li>
<li><b>device 数各用 paper 原生</b>（Duplex=4 / Helios=8 / Stratum-L=6）→ 聚合带宽不对等（同 exp1 AMMA-vs-Rubin caveat）。</li>
<li><b>Stratum tiering 按命中率加权</b>（p_hot=0.9）；仅 expert-read 段享受，近似应用到整段 MoE FFN（expert 占 FFN 字节 &gt;95%）。</li>
<li><b>Duplex MLA 长 CL compute-bound</b>：真实 8:1 compute:BW（crossover OI=8），MLA-absorbed OI≈64–128 ≫ 8 → 长 CL attention 落 compute 边，模型如实暴露。</li>
<li><b>dtype</b>：exp4 三条 FP16，exp1 LPU/Rubin FP8（已在字节数体现）。</li>
</ul>
<p style="color:#888;font-size:12px">Generated by <code>backend/gen_report_html.py</code>. 数据源见各 section 的脚本依赖。</p>
"""


def main():
    html = (f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            f"<title>Decode 加速器实验报告 (fig1/exp1/exp3/exp4)</title>"
            f"<style>{CSS}</style></head><body>"
            + section_overview() + section_fig1() + section_exp1()
            + section_exp3() + section_exp4() + section_caveats()
            + "</body></html>")
    out = os.path.join(HERE, "experiments_report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
