# we only consider Qwen 3 235B model FP8
# We only consider Decodeing stage, all layers.
import math
import hardware_config

config = {
    "d_model": 4096,            # hidden size
    "num_attention_heads": 64,   # query heads
    "num_kv_heads": 4,           # key/value heads (GQA)
    "d_head": 128,               # per-head dimension
    "num_layers": 94,            # transformer layers
}

B200_config = hardware_config.B200_config


ours_config = hardware_config.ours_config

rubin_config = hardware_config.rubin_config

# ---------------------------------------------------------------------------
# Strategy-aware per-NPU model configs for single-layer GQA analysis
#   tp_h=4 groups, tp_s=4 cubes per group, 16 NPUs total
# ---------------------------------------------------------------------------

FULL_CONFIG = config  # alias for the full-model config

def make_strategy_config(strategy, tp_h=4, tp_hd=4):
    """Build per-NPU model config for a given parallelism strategy.

    All three strategies split Wq/Wk/Wv by Hkv (tp_h groups) then by d_head (tp_hd cubes).
    Each cube computes QKV with dk_local = d_head_full // tp_hd = 32.

    HMP/Baseline: AllGather(tp_hd) within group reconstructs dk=128.
      KV cache stored along seq across tp_s cubes (CacheSeq = seq/tp_s).
      Attention uses dk_full=128 on local CacheSeq slice.
    TP16: NO AllGather on QKV.  Attention uses dk=32 on full seq (partial
      inner product).  AllReduce on scores handled by AstraSim comm.

    tp_h:  head-group count  (Hkv split, default 4)
    tp_hd: head-dim split factor within each group (default 4)
    """
    Hq = FULL_CONFIG["num_attention_heads"]
    Hkv = FULL_CONFIG["num_kv_heads"]
    G = tp_h

    if strategy == "hmp":
        return {
            "d_model": FULL_CONFIG["d_model"],
            "num_attention_heads": Hq // G,
            "num_kv_heads": Hkv // G,
            "d_head": FULL_CONFIG["d_head"] // tp_hd,
            "d_head_full": FULL_CONFIG["d_head"],
            "num_layers": 1,
            "tp_s": tp_hd,
            "tp_hd": tp_hd,
            "tp_h": tp_h,
            "wo_mode": "sharded",
        }
    elif strategy == "HMP_reo":
        return {
            "d_model": FULL_CONFIG["d_model"],
            "num_attention_heads": Hq // G,
            "num_kv_heads": Hkv // G,
            "d_head": FULL_CONFIG["d_head"] // tp_hd,
            "d_head_full": FULL_CONFIG["d_head"],
            "num_layers": 1,
            "tp_s": tp_hd,
            "tp_hd": tp_hd,
            "tp_h": tp_h,
            "wo_mode": "duplicate",
        }
    elif strategy == "hmp_reo_new":
        return {
            "d_model": FULL_CONFIG["d_model"],
            "num_attention_heads": Hq // G,
            "num_kv_heads": Hkv // G,
            "d_head": FULL_CONFIG["d_head"] // tp_hd,
            "d_head_full": FULL_CONFIG["d_head"],
            "num_layers": 1,
            "tp_s": tp_hd,
            "tp_hd": tp_hd,
            "tp_h": tp_h,
            "wo_mode": "row_sharded",
        }
    elif strategy == "tp16":
        return {
            "d_model": FULL_CONFIG["d_model"],
            "num_attention_heads": Hq // Hkv,     # 16 query heads per KV group
            "num_kv_heads": Hkv // G,             # 1 KV head per group
            "d_head": FULL_CONFIG["d_head"] // tp_hd,  # 32, HeadDim split by tp_hd
            # NO d_head_full: TP16 does NOT AllGather QKV results.
            # Attention runs with dk=32 (partial inner product, AllReduce on scores).
            "num_layers": 1,
            "tp_s": 1,        # no CacheSeq split (full sequence per NPU)
            "tp_hd": tp_hd,   # d_head split for QKV weight sizing
            "wo_mode": "full",
        }
    elif strategy == "rubin":
        return {
            "d_model": FULL_CONFIG["d_model"],
            "num_attention_heads": Hq,
            "num_kv_heads": Hkv,
            "d_head": FULL_CONFIG["d_head"],
            "num_layers": 1,
            "tp_s": 1,
            "wo_mode": "none",
            "fused_attention": True,  # single GPU fused kernel (FlashDecoding)
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def single_layer_qkv_time(hw, cfg, batch_size, seq):
    """QKV projection time for one decode step, single layer.
    Compute-bound (OI >> 1).  hw['compute'] is TFLOPS.
    FLOPs = 2 * MACs for matrix multiply."""
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]
    Hkv = cfg["num_kv_heads"]
    dk = cfg["d_head"]
    q_mac = batch_size * Hq * dk * d
    kv_mac = batch_size * 2 * Hkv * dk * d
    flops = 2 * (q_mac + kv_mac)
    return flops / (hw["compute"] * 1e12)


def single_layer_attention_time(hw, cfg, batch_size, seq):
    """Attention kernel time for one decode step, single layer.
    hw['compute'] is TFLOPS, hw['Bandwidth'] is TB/s.

    Two modes depending on cfg["fused_attention"]:
      False (default, AstraSim trace model):
        Serial sub-ops as encoded in the Chakra trace:
          2x KV-cache concat (memory-bound)
          2x einsum QK^T / Score@V (compute-bound, FLOPs = 2*MACs)
      True (Rubin fused kernel):
        Single roofline: max(total_memory_time, total_compute_time)

    Uses d_head_full (after AllGather) when available; falls back to d_head.
    """
    Hq = cfg["num_attention_heads"]
    Hkv_cache = cfg.get("num_kv_heads_cache", cfg["num_kv_heads"])
    dk = cfg.get("d_head_full", cfg["d_head"])
    tp_s = cfg.get("tp_s", 1)
    if tp_s > 1:
        cache_seq = seq // tp_s + 1  # HMP/Baseline: Seq/tp_s cached + 1 new
    else:
        cache_seq = seq              # TP16/Rubin: full sequence per NPU

    if cfg.get("fused_attention", False):
        kv_elems = batch_size * 2 * Hkv_cache * cache_seq * dk
        memory_time = kv_elems / (hw["Bandwidth"] * 1e12)
        einsum_mac = 2 * batch_size * Hq * cache_seq * dk
        compute_time = (2 * einsum_mac) / (hw["compute"] * 1e12)
        return max(memory_time, compute_time)

    concat_elem = batch_size * cache_seq * dk * Hkv_cache
    concat_time = concat_elem / (hw["Bandwidth"] * 1e12)

    einsum_mac = batch_size * Hq * cache_seq * dk
    einsum_time = (2 * einsum_mac) / (hw["compute"] * 1e12)

    return 2 * concat_time + 2 * einsum_time


def single_layer_output_time(hw, cfg, batch_size, seq):
    """Output projection time for one decode step, single layer.
    Compute-bound.  hw['compute'] is TFLOPS.  FLOPs = 2 * MACs.

    Uses d_head_full (after AllGather) when available; falls back to d_head.
    """
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]
    dk = cfg.get("d_head_full", cfg["d_head"])
    Dq = Hq * dk
    wo_mode = cfg.get("wo_mode", "duplicate")

    if wo_mode == "duplicate":
        mac = batch_size * Dq * d
    elif wo_mode == "sharded":
        tp_s = cfg.get("tp_s", 1)
        mac = batch_size * (Dq // tp_s) * d
    elif wo_mode == "row_sharded":
        tp_s = cfg.get("tp_s", 1)
        mac = batch_size * Hq * (dk // tp_s) * d
    elif wo_mode == "full":
        mac = batch_size * Dq * d
    else:
        mac = batch_size * Hq * dk * d

    return (2 * mac) / (hw["compute"] * 1e12)


def single_layer_total_time(hw, cfg, batch_size, seq):
    """Total per-NPU time for single GQA layer (QKV + Attn + Output)."""
    return (single_layer_qkv_time(hw, cfg, batch_size, seq)
            + single_layer_attention_time(hw, cfg, batch_size, seq)
            + single_layer_output_time(hw, cfg, batch_size, seq))


def qkv_allgather_comm_time(hw, cfg, batch_size,
                             hop_latency_ns=100, endpoint_delay_ns=10,
                             hops_per_step=1):
    """AllGather(tp_hd) within cube group after QKV projection (HeadDim split).

    Each cube holds Q[B, Hq, dk_local], K[B, Hkv, dk_local], V[B, Hkv, dk_local]
    where dk_local = d_head_full // tp_hd.  Ring AllGather across tp_hd cubes
    reconstructs full d_head_full for attention.
    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    Returns time in seconds.  Zero if no d_head split.
    """
    tp_hd = cfg.get("tp_hd", 1)
    d_head_full = cfg.get("d_head_full")
    if tp_hd <= 1 or d_head_full is None:
        return 0.0
    Hq = cfg["num_attention_heads"]
    Hkv = cfg["num_kv_heads"]
    dk_local = cfg["d_head"]
    chunk_elems = batch_size * (Hq + 2 * Hkv) * dk_local
    link_bw = hw.get("device_link_bw", 2.0) * 1e12
    n_steps = tp_hd - 1
    bw_per_step = chunk_elems / link_bw
    fixed_per_step = (hops_per_step * hop_latency_ns + endpoint_delay_ns) * 1e-9
    return n_steps * (fixed_per_step + bw_per_step)


def baseline_output_allgather_comm_time(hw, cfg, batch_size,
                                        hop_latency_ns=100,
                                        endpoint_delay_ns=10,
                                        hops_per_step=1):
    """AllGather(tp_s) for Baseline's sharded output projection (wo).

    After AllReduce on tp_h, each cube holds [B, D/tp_s].
    Ring AllGather across tp_s cubes reconstructs full [B, D].
    block2x2 layout: g21 groups are all 1-hop, no congestion.
    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    Returns time in seconds.  Zero if wo is not sharded or no tp_s split.
    """
    tp_s = cfg.get("tp_s", 1)
    wo_mode = cfg.get("wo_mode", "duplicate")
    if tp_s <= 1 or wo_mode != "sharded":
        return 0.0
    n_steps = tp_s - 1
    chunk_elems = batch_size * cfg["d_model"] // tp_s
    link_bw = hw.get("device_link_bw", 2.0) * 1e12
    bw_per_step = chunk_elems / link_bw
    fixed_per_step = (hops_per_step * hop_latency_ns + endpoint_delay_ns) * 1e-9
    return n_steps * (fixed_per_step + bw_per_step)


def hmp_reo_new_attn_rs_comm_time(hw, cfg, batch_size,
                                   hop_latency_ns=100, endpoint_delay_ns=10,
                                   hops_per_step=1):
    """ReduceScatter(tp_s) on attention output for hmp_reo_new strategy.

    After attention kernel, each cube holds the full HeadDim partial-sum
    output [B, Hq/tp_h, Head/KVHead, 1, d_head_full] with tp_s partial sums.
    Ring ReduceScatter across tp_s cubes reduces and scatters along HeadDim,
    giving each cube [B, Hq/tp_h, Head/KVHead, 1, d_head_full/tp_s].
    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    Returns time in seconds.  Zero if no tp_s split.
    """
    tp_s = cfg.get("tp_s", 1)
    if tp_s <= 1:
        return 0.0
    Hq = cfg["num_attention_heads"]
    dk = cfg.get("d_head_full", cfg["d_head"])
    total_elems = batch_size * Hq * dk
    chunk_elems = total_elems // tp_s
    link_bw = hw.get("device_link_bw", 2.0) * 1e12
    n_steps = tp_s - 1
    bw_per_step = chunk_elems / link_bw
    fixed_per_step = (hops_per_step * hop_latency_ns + endpoint_delay_ns) * 1e-9
    return n_steps * (fixed_per_step + bw_per_step)


def hmp_final_reduce_comm_time(hw, cfg, batch_size,
                                hop_latency_ns=100, endpoint_delay_ns=10,
                                hops_per_step=1):
    """Tree-reduce across all tp_h × tp_s NPUs so one cube gets the complete
    GQA output [B, d_model].

    log2(tp_h * tp_s) steps, each step transfers B × d_model elements.
    Per-step time = hops * hop_latency + endpoint_delay + msg/link_bw.
    Returns time in seconds.  Zero for single-NPU strategies.
    """
    tp_h = cfg.get("tp_h", 1)
    tp_s = cfg.get("tp_s", 1)
    n_total = tp_h * tp_s
    if n_total <= 1:
        return 0.0
    num_steps = int(math.log2(n_total))
    msg_elems = batch_size * cfg["d_model"]
    link_bw = hw.get("device_link_bw", 2.0) * 1e12
    bw_per_step = msg_elems / link_bw
    fixed_per_step = (hops_per_step * hop_latency_ns + endpoint_delay_ns) * 1e-9
    return num_steps * (fixed_per_step + bw_per_step)


# TODO: SIM_DATA below was collected with old TMAC/s convention (peak-perf
# interpreted as TMAC/s). After rebuilding AstraSim with the TFLOPS fix
# (2*MACs in Workload.cc), re-run simulations to update these values.
SIM_DATA = {
    1024: {
        64: {
            "HMP_reo":      {"gpu": 42064, "qkv": 20131.0,"attn": 3928.0,   "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 28538, "qkv": 20131.0,"attn": 3928.0,   "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 13417, "qkv": 5032.0, "attn": 3912.0,   "output": 4473.0, "comm": 49473},
            "rubin":    {"gpu": 3575,  "qkv": 276.1,  "attn": 3053.4,   "output": 245.4,  "comm": 0},
        },
    },
    2048: {
        64: {
            "HMP_reo":      {"gpu": 45978, "qkv": 20131.0,"attn": 7842.0,   "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 32452, "qkv": 20131.0,"attn": 7842.0,   "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 17331, "qkv": 5032.0, "attn": 7826.0,   "output": 4473.0, "comm": 51521},
            "rubin":    {"gpu": 6625,  "qkv": 276.1,  "attn": 6103.8,   "output": 245.4,  "comm": 0},
        },
    },
    4096: {
        64: {
            "HMP_reo":      {"gpu": 53808, "qkv": 20131.0,"attn": 15672.0,  "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 40282, "qkv": 20131.0,"attn": 15672.0,  "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 25161, "qkv": 5032.0, "attn": 15656.0,  "output": 4473.0, "comm": 55352},
            "rubin":    {"gpu": 12726, "qkv": 276.1,  "attn": 12204.6,  "output": 245.4,  "comm": 0},
        },
    },
    8192: {
        64: {
            "HMP_reo":      {"gpu": 69468, "qkv": 20131.0,"attn": 31332.0,  "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 55942, "qkv": 20131.0,"attn": 31332.0,  "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 40819, "qkv": 5032.0, "attn": 31314.0,  "output": 4473.0, "comm": 63207},
            "rubin":    {"gpu": 24928, "qkv": 276.1,  "attn": 24406.2,  "output": 245.4,  "comm": 0},
        },
    },
    16384: {
        64: {
            "HMP_reo":      {"gpu": 100786,"qkv": 20131.0,"attn": 62650.0,  "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 87260, "qkv": 20131.0,"attn": 62650.0,  "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 72137, "qkv": 5032.0, "attn": 62632.0,  "output": 4473.0, "comm": 82098},
            "rubin":    {"gpu": 49331, "qkv": 276.1,  "attn": 48809.4,  "output": 245.4,  "comm": 0},
        },
    },
    32768: {
        64: {
            "HMP_reo":      {"gpu": 163418,"qkv": 20131.0,"attn": 125282.0, "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 149892,"qkv": 20131.0,"attn": 125282.0, "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 134773,"qkv": 5032.0, "attn": 125268.0, "output": 4473.0, "comm": 133578},
            "rubin":    {"gpu": 98137, "qkv": 276.1,  "attn": 97615.9,  "output": 245.4,  "comm": 0},
        },
    },
    65536: {
        1: {
            "HMP_reo":      {"gpu": 4505,  "qkv": 313.0,  "attn": 3912.0,    "output": 279.0,  "comm": 0},
            "hmp": {"gpu": 4294,  "qkv": 313.0,  "attn": 3912.0,    "output": 69.0,   "comm": 18240},
            "tp16":     {"gpu": 4058,  "qkv": 69.0,   "attn": 3912.0,    "output": 69.0,   "comm": 49063},
            "rubin":    {"gpu": 3059,  "qkv": 4.3,    "attn": 3050.4,    "output": 3.8,    "comm": 0},
        },
        2: {
            "HMP_reo":      {"gpu": 9018,  "qkv": 628.0,  "attn": 7828.0,    "output": 559.0,  "comm": 0},
            "hmp": {"gpu": 8595,  "qkv": 628.0,  "attn": 7828.0,    "output": 139.0,  "comm": 18240},
            "tp16":     {"gpu": 8121,  "qkv": 156.0,  "attn": 7826.0,    "output": 139.0,  "comm": 51111},
            "rubin":    {"gpu": 6117,  "qkv": 8.6,    "attn": 6100.9,    "output": 7.7,    "comm": 0},
        },
        4: {
            "HMP_reo":      {"gpu": 18039, "qkv": 1257.0, "attn": 15658.0,   "output": 1118.0, "comm": 0},
            "hmp": {"gpu": 17194, "qkv": 1257.0, "attn": 15658.0,   "output": 279.0,  "comm": 18240},
            "tp16":     {"gpu": 16248, "qkv": 313.0,  "attn": 15656.0,   "output": 279.0,  "comm": 54942},
            "rubin":    {"gpu": 12234, "qkv": 17.3,   "attn": 12201.8,   "output": 15.3,   "comm": 0},
        },
        8: {
            "HMP_reo":      {"gpu": 36082, "qkv": 2515.0, "attn": 31318.0,   "output": 2249.0, "comm": 0},
            "hmp": {"gpu": 34392, "qkv": 2515.0, "attn": 31318.0,   "output": 559.0,  "comm": 18240},
            "tp16":     {"gpu": 32501, "qkv": 628.0,  "attn": 31314.0,   "output": 559.0,  "comm": 62797},
            "rubin":    {"gpu": 24469, "qkv": 34.5,   "attn": 24403.6,   "output": 30.7,   "comm": 0},
        },
        16: {
            "HMP_reo":      {"gpu": 72167, "qkv": 5032.0, "attn": 62636.0,   "output": 4499.0, "comm": 0},
            "hmp": {"gpu": 68786, "qkv": 5032.0, "attn": 62636.0,   "output": 1118.0, "comm": 18240},
            "tp16":     {"gpu": 65007, "qkv": 1257.0, "attn": 62632.0,   "output": 1118.0, "comm": 81688},
            "rubin":    {"gpu": 48938, "qkv": 69.0,   "attn": 48807.2,   "output": 61.4,   "comm": 0},
        },
        32: {
            "HMP_reo":      {"gpu": 144342,"qkv": 10065.0,"attn": 125276.0,  "output": 8999.0, "comm": 0},
            "hmp": {"gpu": 137579,"qkv": 10065.0,"attn": 125276.0,  "output": 2236.0, "comm": 18240},
            "tp16":     {"gpu": 130019,"qkv": 2515.0, "attn": 125268.0,  "output": 2236.0, "comm": 133168},
            "rubin":    {"gpu": 97875, "qkv": 138.1,  "attn": 97614.4,   "output": 122.7,  "comm": 0},
        },
        64: {
            "HMP_reo":      {"gpu": 288690,"qkv": 20131.0,"attn": 250554.0,  "output": 17999.0,"comm": 0},
            "hmp": {"gpu": 275164,"qkv": 20131.0,"attn": 250554.0,  "output": 4473.0, "comm": 18265},
            "tp16":     {"gpu": 260043,"qkv": 5032.0, "attn": 250538.0,  "output": 4473.0, "comm": 236538},
            "rubin":    {"gpu": 195750,"qkv": 276.1,  "attn": 195228.8,  "output": 245.4,  "comm": 0},
        },
    },
}


def verify_against_sim(batch_sizes=None, seq=65536):
    """Print predicted vs simulation values for all strategies and batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 64]
    hbm4 = hardware_config.hbm4_npu_config
    rubin_hw = hardware_config.rubin_single_layer_config
    strategies = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]

    for bs in batch_sizes:
        print(f"{'=' * 80}")
        print(f"  Batch={bs}, Seq={seq}")
        print(f"{'=' * 80}")
        sd_batch = SIM_DATA.get(seq, {}).get(bs, {})

        header = f"  {'Strategy':<14} {'Module':<8} {'Predicted':>12}  {'Sim':>12}  {'Err':>7}  {'Bound'}"
        print(header)
        print(f"  {'-'*79}")

        for strategy in strategies:
            cfg = make_strategy_config(strategy)
            hw = rubin_hw if strategy == "rubin" else hbm4
            sd = sd_batch.get(strategy, {})

            qkv_ns = single_layer_qkv_time(hw, cfg, bs, seq) * 1e9
            attn_ns = single_layer_attention_time(hw, cfg, bs, seq) * 1e9
            out_ns = single_layer_output_time(hw, cfg, bs, seq) * 1e9
            total_ns = qkv_ns + attn_ns + out_ns

            for label, pred, key in [("QKV", qkv_ns, "qkv"),
                                      ("Attn", attn_ns, "attn"),
                                      ("Output", out_ns, "output")]:
                sim_val = sd.get(key, 0)
                err = abs(pred - sim_val) / max(sim_val, 1e-9) * 100 if sim_val else 0
                bound = "comp" if label != "Attn" else "mixed"
                tag = f"[{strategy.upper()}]" if label == "QKV" else ""
                print(f"  {tag:<14} {label:<8} {pred:>10.1f}ns  {sim_val:>10.1f}ns  {err:>5.1f}%   {bound}")

            sim_gpu = sd.get("gpu", 0)
            sim_comm = sd.get("comm", 0)
            sim_wall = sim_gpu + sim_comm

            if strategy in ("hmp_reo_new", "HMP_reo"):
                ana_comm = hmp_final_reduce_comm_time(hw, cfg, bs) * 1e9
                if strategy == "hmp_reo_new":
                    ana_comm += hmp_reo_new_attn_rs_comm_time(hw, cfg, bs) * 1e9
                print(f"  {'':14} {'Total':<8} {total_ns:>10.1f}ns  {sim_gpu:>10.1f}ns(gpu)  "
                      f"ana_comm={ana_comm:.0f}ns  pred_wall={total_ns + ana_comm:.0f}ns")
            else:
                print(f"  {'':14} {'Total':<8} {total_ns:>10.1f}ns  {sim_gpu:>10.1f}ns(gpu)  "
                      f"comm={sim_comm}ns  wall={sim_wall}ns")
            print()

        print()


def generate_comparison_json(output_dir="reports"):
    """Generate JSON files comparing predicted (gpu+comm) vs sim for all strategies.

    Predicted gpu_ns comes from the analytical model.
    comm_ns comes from AstraSim simulation data (strategy/topology dependent).
    predicted wall_ns = predicted_gpu + sim_comm.
    """
    import json, os
    os.makedirs(output_dir, exist_ok=True)

    hbm4 = hardware_config.hbm4_npu_config
    rubin_hw = hardware_config.rubin_single_layer_config
    strategies = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]

    batch_scaling_json = _build_scaling_json(
        strategies, hbm4, rubin_hw, "batch",
        seq=65536, vary_values=[1, 2, 4, 8, 16, 32, 64])

    seq_scaling_json = _build_scaling_json(
        strategies, hbm4, rubin_hw, "seq",
        batch=64, vary_values=[1024, 2048, 4096, 8192, 16384, 32768, 65536])

    batch_path = os.path.join(output_dir, "predicted_batch_scaling_sl65536.json")
    with open(batch_path, "w") as f:
        json.dump(batch_scaling_json, f, indent=2)
    print(f"Saved: {batch_path}")

    seq_path = os.path.join(output_dir, "predicted_seq_scaling_bs64.json")
    with open(seq_path, "w") as f:
        json.dump(seq_scaling_json, f, indent=2)
    print(f"Saved: {seq_path}")


def _build_scaling_json(strategies, hbm4, rubin_hw, mode, **kwargs):
    """Build comparison dict for one scaling dimension.

    mode="batch": fix seq, vary batch.  mode="seq": fix batch, vary seq.
    """
    result = {}
    for strategy in strategies:
        cfg = make_strategy_config(strategy)
        hw = rubin_hw if strategy == "rubin" else hbm4
        entries = []

        for val in kwargs["vary_values"]:
            if mode == "batch":
                bs, seq = val, kwargs["seq"]
            else:
                bs, seq = kwargs["batch"], val

            qkv_ns = single_layer_qkv_time(hw, cfg, bs, seq) * 1e9
            attn_ns = single_layer_attention_time(hw, cfg, bs, seq) * 1e9
            out_ns = single_layer_output_time(hw, cfg, bs, seq) * 1e9
            pred_gpu = qkv_ns + attn_ns + out_ns

            sd = SIM_DATA.get(seq, {}).get(bs, {}).get(strategy, {})
            sim_gpu = sd.get("gpu", 0)
            sim_comm = sd.get("comm", 0)
            sim_wall = sim_gpu + sim_comm

            if strategy in ("hmp_reo_new", "HMP_reo"):
                ana_comm = hmp_final_reduce_comm_time(hw, cfg, bs) * 1e9
                if strategy == "hmp_reo_new":
                    ana_comm += hmp_reo_new_attn_rs_comm_time(hw, cfg, bs) * 1e9
                pred_comm = round(ana_comm, 1)
            else:
                pred_comm = sim_comm

            pred_wall = pred_gpu + pred_comm

            entry = {
                "batch" if mode == "batch" else "seq": val,
                "predicted": {
                    "gpu_ns": round(pred_gpu, 1),
                    "comm_ns": pred_comm,
                    "wall_ns": round(pred_wall, 1),
                    "modules": {
                        "qkv_comp": round(qkv_ns, 1),
                        "attn_comp": round(attn_ns, 1),
                        "output_comp": round(out_ns, 1),
                    },
                },
                "sim": {
                    "gpu_ns": sim_gpu,
                    "comm_ns": sim_comm,
                    "wall_ns": sim_wall,
                    "modules": sd,
                },
                "error": {
                    "gpu_pct": round(abs(pred_gpu - sim_gpu) / max(sim_gpu, 1) * 100, 2) if sim_gpu else 0,
                    "wall_pct": round(abs(pred_wall - sim_wall) / max(sim_wall, 1) * 100, 2) if sim_wall else 0,
                },
            }
            entries.append(entry)

        label_map = {
            "HMP_reo": "HMP_reo (block2x2 Mesh2D)",
            "hmp_reo_new": "HMP_reo_new (Mesh2D, RS+TreeReduce)",
            "hmp": "HMP (block2x2 Mesh2D)",
            "tp16": "TP16 (Torus OneRing)",
            "rubin": "Rubin (1GPU)",
        }
        result[strategy] = {
            "label": label_map[strategy],
            "data": entries,
        }
    return result


if __name__ == "__main__":
    verify_against_sim(batch_sizes=[1, 64])
    print()
    generate_comparison_json()

# compute cost for QKV projection for single step in decode
def calculate_compute_cost_qkv(config, batch_size, t):
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    compute_cost = 0
    # WqX: (B, 1, d_model) @ (d_model, num_attention_heads * d_head)
    compute_cost += 2 * batch_size * 1 * d_model * (num_attention_heads * d_head)
    # WkX: (B, 1, d_model) @ (d_model, num_kv_heads * d_head)
    compute_cost += 2 * batch_size * 1 * d_model * (num_kv_heads * d_head)
    # WvX: (B, 1, d_model) @ (d_model, num_kv_heads * d_head)
    compute_cost += 2 * batch_size * 1 * d_model * (num_kv_heads * d_head)
    return compute_cost * num_layers

# compute cost for O projection for single step in decode
def calculate_compute_cost_o(config, batch_size, t):
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    # X'Wo: (B, 1, num_attention_heads * d_head) @ (num_attention_heads * d_head, d_model)
    compute_cost = 2 * batch_size * 1 * (num_attention_heads * d_head) * d_model
    return compute_cost * num_layers

# compute cost for projection (QKV + O) for single step in decode
def calculate_compute_cost_projection(config, batch_size, t):
    return calculate_compute_cost_qkv(config, batch_size, t) + calculate_compute_cost_o(config, batch_size, t)

# compute cost for attention for single step in decode
def calculate_compute_cost_attention(config, batch_size, t):
    # t means the current decode step (number of past tokens).
    # We generate 1 new token: query is (B, 1, ...), keys/values span t positions.
    # Computed across all layers.
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    compute_cost = 0
    # QK^T: (B, num_attention_heads, 1, d_head) @ (B, num_attention_heads, d_head, t)
    compute_cost += 2 * batch_size * num_attention_heads * 1 * t * d_head
    # softmax over t scores per head
    compute_cost += 5 * batch_size * num_attention_heads * 1 * t
    # score * V: (B, num_attention_heads, 1, t) @ (B, num_attention_heads, t, d_head)
    compute_cost += 2 * batch_size * num_attention_heads * 1 * t * d_head
    # multiply by num_layers (each layer has independent attention)
    return compute_cost * num_layers

# memory bytes transfer for QKV projection for single step in decode
def calculate_memory_bytes_transfer_qkv(config, batch_size, t):
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    memory_bytes_transfer = 0
    # input activation read x (B, 1, d_model)
    memory_bytes_transfer += batch_size * 1 * d_model
    # Q projection, read Wq
    memory_bytes_transfer += d_model * num_attention_heads * d_head
    # K projection, read Wk
    memory_bytes_transfer += d_model * num_kv_heads * d_head
    # V projection, read Wv
    memory_bytes_transfer += d_model * num_kv_heads * d_head
    return memory_bytes_transfer * num_layers

# memory bytes transfer for O projection for single step in decode
def calculate_memory_bytes_transfer_o(config, batch_size, t):
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    memory_bytes_transfer = 0
    # output projection, read Wo
    memory_bytes_transfer += d_model * num_attention_heads * d_head
    # write output back
    memory_bytes_transfer += batch_size * d_model
    return memory_bytes_transfer * num_layers

# memory bytes transfer for projection (QKV + O) for single step in decode
def calculate_memory_bytes_transfer_projection(config, batch_size, t):
    return calculate_memory_bytes_transfer_qkv(config, batch_size, t) + calculate_memory_bytes_transfer_o(config, batch_size, t)

# memory bytes transfer for attention for single step in decode
def calculate_memory_bytes_transfer_attention(config, batch_size, t):
    d_model = config["d_model"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    memory_bytes_transfer = 0
    # kv cache read k, v: (B, num_kv_heads, t, d_head) x 2
    memory_bytes_transfer += batch_size * 2 * num_kv_heads * d_head * t
    # for each decode step, kv cache write 1 new token
    memory_bytes_transfer += batch_size * 2 * num_kv_heads * d_head
    return memory_bytes_transfer * num_layers


# memory latency for QKV projection for single step in decode
def calculate_memory_latency_qkv(hardware_config, config, batch_size, t):
    memory_bytes = calculate_memory_bytes_transfer_qkv(config, batch_size, t)
    return memory_bytes / ((hardware_config["Bandwidth"] * hardware_config["num_devices"]) * 1e12)

# memory latency for O projection for single step in decode
def calculate_memory_latency_o(hardware_config, config, batch_size, t):
    memory_bytes = calculate_memory_bytes_transfer_o(config, batch_size, t)
    return memory_bytes / ((hardware_config["Bandwidth"] * hardware_config["num_devices"]) * 1e12)

# memory latency for projection (QKV + O) for single step in decode
def calculate_memory_latency_projection(hardware_config, config, batch_size, t):
    return calculate_memory_latency_qkv(hardware_config, config, batch_size, t) + calculate_memory_latency_o(hardware_config, config, batch_size, t)

# memory latency for attention for single step in decode
def calculate_memory_latency_attention(hardware_config, config, batch_size, t):
    memory_bytes = calculate_memory_bytes_transfer_attention(config, batch_size, t)
    return memory_bytes / ((hardware_config["Bandwidth"] * hardware_config["num_devices"]) * 1e12)

def calculate_memory_total_latency_qkv(hardware_config, config, batch_size, t):
    total_latency = 0
    for i in range(t):
        total_latency += calculate_memory_latency_qkv(hardware_config, config, batch_size, i)
    return total_latency

def calculate_memory_total_latency_o(hardware_config, config, batch_size, t):
    total_latency = 0
    for i in range(t):
        total_latency += calculate_memory_latency_o(hardware_config, config, batch_size, i)
    return total_latency

def calculate_memory_total_latency_projection(hardware_config, config, batch_size, t):
    total_latency = 0
    for i in range(t):
        total_latency += calculate_memory_latency_projection(hardware_config, config, batch_size, i)
    return total_latency

def calculate_memory_total_latency_attention(hardware_config, config, batch_size, t):
    total_latency = 0
    for i in range(t):
        total_latency += calculate_memory_latency_attention(hardware_config, config, batch_size, i)
    return total_latency

# kv cache full check
def is_kv_cache_full(hardware_config, config, batch_size, t):
    '''
    given the batch size and sequence length, check if the memory capacity is enough.
    '''
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    kv_bytes_per_layer = batch_size * 2 * num_kv_heads * d_head * t
    kv_write_per_layer = batch_size * 2 * num_kv_heads * d_head
    total_kv_bytes = (kv_bytes_per_layer + kv_write_per_layer) * num_layers
    if total_kv_bytes <= hardware_config["kv_cache_size"] * 1e9:
        return False
    else:
        return True

def calculate_total_kv_bytes(config, batch_size, t):
    num_kv_heads = config["num_kv_heads"]
    d_head = config["d_head"]
    num_layers = config["num_layers"]
    kv_bytes_per_layer = batch_size * 2 * num_kv_heads * d_head * t
    kv_write_per_layer = batch_size * 2 * num_kv_heads * d_head
    total_kv_bytes = (kv_bytes_per_layer + kv_write_per_layer) * num_layers
    return total_kv_bytes

# compute time for QKV projection for single step in decode
def calculate_compute_time_qkv(hardware_config, config, batch_size, t):
    Flops = calculate_compute_cost_qkv(config, batch_size, t)
    return Flops / ((hardware_config["compute"] * hardware_config["num_devices"]) * 1e12)

# compute time for O projection for single step in decode
def calculate_compute_time_o(hardware_config, config, batch_size, t):
    Flops = calculate_compute_cost_o(config, batch_size, t)
    return Flops / ((hardware_config["compute"] * hardware_config["num_devices"]) * 1e12)

# compute time for projection (QKV + O) for single step in decode
def calculate_compute_time_projection(hardware_config, config, batch_size, t):
    return calculate_compute_time_qkv(hardware_config, config, batch_size, t) + calculate_compute_time_o(hardware_config, config, batch_size, t)

# compute time for attention for single step in decode
def calculate_compute_time_attention(hardware_config, config, batch_size, t):
    Flops = calculate_compute_cost_attention(config, batch_size, t)
    return Flops / ((hardware_config["compute"] * hardware_config["num_devices"]) * 1e12)

# overall time (3 sequential stages: QKV -> attention -> O)
def calculate_total_time(hardware_config, config, batch_size, t):
    total_time = 0
    for i in range(t):
        total_time += max(calculate_compute_time_qkv(hardware_config, config, batch_size, i), calculate_memory_latency_qkv(hardware_config, config, batch_size, i))
        total_time += max(calculate_compute_time_attention(hardware_config, config, batch_size, i), calculate_memory_latency_attention(hardware_config, config, batch_size, i))
        total_time += max(calculate_compute_time_o(hardware_config, config, batch_size, i), calculate_memory_latency_o(hardware_config, config, batch_size, i))
    return total_time

# overall time for QKV projection
def calculate_total_time_qkv(hardware_config, config, batch_size, t):
    total_time = 0
    for i in range(t):
        total_time += max(calculate_compute_time_qkv(hardware_config, config, batch_size, i), calculate_memory_latency_qkv(hardware_config, config, batch_size, i))
    return total_time

# overall time for O projection
def calculate_total_time_o(hardware_config, config, batch_size, t):
    total_time = 0
    for i in range(t):
        total_time += max(calculate_compute_time_o(hardware_config, config, batch_size, i), calculate_memory_latency_o(hardware_config, config, batch_size, i))
    return total_time

# overall time for projection (QKV + O)
def calculate_total_time_projection(hardware_config, config, batch_size, t):
    return calculate_total_time_qkv(hardware_config, config, batch_size, t) + calculate_total_time_o(hardware_config, config, batch_size, t)

# overall time for attention
def calculate_total_time_attention(hardware_config, config, batch_size, t):
    total_time = 0
    for i in range(t):
        total_time += max(calculate_compute_time_attention(hardware_config, config, batch_size, i), calculate_memory_latency_attention(hardware_config, config, batch_size, i))
    return total_time

# overall compute time for QKV projection
def calculate_compute_total_time_qkv(hardware_config, config, batch_size, t):
    compute_time = 0
    for i in range(t):
        compute_time += calculate_compute_time_qkv(hardware_config, config, batch_size, i)
    return compute_time

# overall compute time for O projection
def calculate_compute_total_time_o(hardware_config, config, batch_size, t):
    compute_time = 0
    for i in range(t):
        compute_time += calculate_compute_time_o(hardware_config, config, batch_size, i)
    return compute_time

# overall compute time for projection (QKV + O)
def calculate_compute_total_time_projection(hardware_config, config, batch_size, t):
    return calculate_compute_total_time_qkv(hardware_config, config, batch_size, t) + calculate_compute_total_time_o(hardware_config, config, batch_size, t)

# overall compute time for attention
def calculate_compute_total_time_attention(hardware_config, config, batch_size, t):
    compute_time = 0
    for i in range(t):
        compute_time += calculate_compute_time_attention(hardware_config, config, batch_size, i)
    return compute_time


'''
    compute utilization = total Flops / (time * compute power)
    bandwidth utilization = total bytes / (time * bandwidth)
'''
def calculate_compute_util_qkv(hardware_config, config, batch_size, t):
    total_flops = 0
    time = calculate_total_time_qkv(hardware_config, config, batch_size, t)
    for i in range(t):
        total_flops += calculate_compute_cost_qkv(config, batch_size, i)
    return total_flops / (time * hardware_config["compute"] * hardware_config["num_devices"] * 1e12)

def calculate_compute_util_o(hardware_config, config, batch_size, t):
    total_flops = 0
    time = calculate_total_time_o(hardware_config, config, batch_size, t)
    for i in range(t):
        total_flops += calculate_compute_cost_o(config, batch_size, i)
    return total_flops / (time * hardware_config["compute"] * hardware_config["num_devices"] * 1e12)

def calculate_compute_util_projection(hardware_config, config, batch_size, t):
    total_flops = 0
    time = calculate_total_time_projection(hardware_config, config, batch_size, t)
    for i in range(t):
        total_flops += calculate_compute_cost_projection(config, batch_size, i)
    return total_flops / (time * hardware_config["compute"] * hardware_config["num_devices"] * 1e12)

def calculate_compute_util_attention(hardware_config, config, batch_size, t):
    total_flops = 0
    time = calculate_total_time_attention(hardware_config, config, batch_size, t)
    for i in range(t):
        total_flops += calculate_compute_cost_attention(config, batch_size, i)
    return total_flops / (time * hardware_config["compute"] * hardware_config["num_devices"] * 1e12)

def calculate_memory_util_qkv(hardware_config, config, batch_size, t):
    total_bytes = 0
    time = calculate_total_time_qkv(hardware_config, config, batch_size, t)
    for i in range(t):
        total_bytes += calculate_memory_bytes_transfer_qkv(config, batch_size, i)
    return total_bytes / (time * hardware_config["Bandwidth"] * hardware_config["num_devices"] * 1e12)

def calculate_memory_util_o(hardware_config, config, batch_size, t):
    total_bytes = 0
    time = calculate_total_time_o(hardware_config, config, batch_size, t)
    for i in range(t):
        total_bytes += calculate_memory_bytes_transfer_o(config, batch_size, i)
    return total_bytes / (time * hardware_config["Bandwidth"] * hardware_config["num_devices"] * 1e12)

def calculate_memory_util_projection(hardware_config, config, batch_size, t):
    total_bytes = 0
    time = calculate_total_time_projection(hardware_config, config, batch_size, t)
    for i in range(t):
        total_bytes += calculate_memory_bytes_transfer_projection(config, batch_size, i)
    return total_bytes / (time * hardware_config["Bandwidth"] * hardware_config["num_devices"] * 1e12)

def calculate_memory_util_attention(hardware_config, config, batch_size, t):
    total_bytes = 0
    time = calculate_total_time_attention(hardware_config, config, batch_size, t)
    for i in range(t):
        total_bytes += calculate_memory_bytes_transfer_attention(config, batch_size, i)
    return total_bytes / (time * hardware_config["Bandwidth"] * hardware_config["num_devices"] * 1e12)

def calculate_compute_util_overall(hardware_config, config, batch_size, t):
    total_flops = 0
    time = calculate_total_time(hardware_config, config, batch_size, t)
    for i in range(t):
        total_flops += calculate_compute_cost_qkv(config, batch_size, i)
        total_flops += calculate_compute_cost_attention(config, batch_size, i)
        total_flops += calculate_compute_cost_o(config, batch_size, i)
    return total_flops / (time * hardware_config["compute"] * hardware_config["num_devices"] * 1e12)

def calculate_memory_util_overall(hardware_config, config, batch_size, t):
    total_bytes = 0
    time = calculate_total_time(hardware_config, config, batch_size, t)
    for i in range(t):
        total_bytes += calculate_memory_bytes_transfer_qkv(config, batch_size, i)
        total_bytes += calculate_memory_bytes_transfer_attention(config, batch_size, i)
        total_bytes += calculate_memory_bytes_transfer_o(config, batch_size, i)
    return total_bytes / (time * hardware_config["Bandwidth"] * hardware_config["num_devices"] * 1e12)

'''
    throughput = batch_size * sequence_length / total_time
'''

def calculate_throughput_qkv(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time_qkv(hardware_config, config, batch_size, sequence_length)
    return batch_size * sequence_length / total_time

def calculate_throughput_o(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time_o(hardware_config, config, batch_size, sequence_length)
    return batch_size * sequence_length / total_time

def calculate_throughput_projection(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time_projection(hardware_config, config, batch_size, sequence_length)
    return batch_size * sequence_length / total_time

def calculate_throughput_attention(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time_attention(hardware_config, config, batch_size, sequence_length)
    return batch_size * sequence_length / total_time

def calculate_throughput_overall(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time(hardware_config, config, batch_size, sequence_length)
    return batch_size * sequence_length / total_time

'''
    throughput per watt = throughput / (num_devices * power)
'''

def calculate_throughput_per_watt_qkv(hardware_config, config, batch_size, sequence_length):
    return calculate_throughput_qkv(hardware_config, config, batch_size, sequence_length) / (hardware_config["num_devices"] * hardware_config["power"])

def calculate_throughput_per_watt_o(hardware_config, config, batch_size, sequence_length):
    return calculate_throughput_o(hardware_config, config, batch_size, sequence_length) / (hardware_config["num_devices"] * hardware_config["power"])

def calculate_throughput_per_watt_projection(hardware_config, config, batch_size, sequence_length):
    return calculate_throughput_projection(hardware_config, config, batch_size, sequence_length) / (hardware_config["num_devices"] * hardware_config["power"])

def calculate_throughput_per_watt_attention(hardware_config, config, batch_size, sequence_length):
    return calculate_throughput_attention(hardware_config, config, batch_size, sequence_length) / (hardware_config["num_devices"] * hardware_config["power"])

def calculate_throughput_per_watt_overall(hardware_config, config, batch_size, sequence_length):
    return calculate_throughput_overall(hardware_config, config, batch_size, sequence_length) / (hardware_config["num_devices"] * hardware_config["power"])

def attention_projection_ratio(hardware_config, config, batch_size, sequence_length):
    total_time = calculate_total_time(hardware_config, config, batch_size, sequence_length)
    attention_time = calculate_total_time_attention(hardware_config, config, batch_size, sequence_length)
    projection_time = calculate_total_time_projection(hardware_config, config, batch_size, sequence_length)
    return attention_time / total_time, projection_time / total_time

def calculate_total_compute_cost_projection(hardware_config, config, batch_size, sequence_length):
    total_compute_cost = 0
    for i in range(sequence_length):
        total_compute_cost += calculate_compute_cost_qkv(config, batch_size, i)
        total_compute_cost += calculate_compute_cost_o(config, batch_size, i)
    return total_compute_cost

def calculate_total_compute_cost_attention(hardware_config, config, batch_size, sequence_length):
    total_compute_cost = 0
    for i in range(sequence_length):
        total_compute_cost += calculate_compute_cost_attention(config, batch_size, i)
    return total_compute_cost

def calculate_total_memory_bytes_transfer_projection(hardware_config, config, batch_size, sequence_length):
    total_memory_bytes_transfer = 0
    for i in range(sequence_length):
        total_memory_bytes_transfer += calculate_memory_bytes_transfer_qkv(config, batch_size, i)
        total_memory_bytes_transfer += calculate_memory_bytes_transfer_o(config, batch_size, i)
    return total_memory_bytes_transfer

def calculate_total_memory_bytes_transfer_attention(hardware_config, config, batch_size, sequence_length):
    total_memory_bytes_transfer = 0
    for i in range(sequence_length):
        total_memory_bytes_transfer += calculate_memory_bytes_transfer_attention(config, batch_size, i)
    return total_memory_bytes_transfer