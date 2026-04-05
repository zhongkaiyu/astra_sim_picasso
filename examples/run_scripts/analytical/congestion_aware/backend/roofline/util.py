import sys
import json
from main import (
    config,
    config_1,
    calculate_sa_qkv_utilization,
    calculate_sa_score_utilization,
    calculate_sa_att_utilization,
    calculate_sa_proj_o_utilization,
)

config_deepseek3 = {
    "d_model": 7168,
    "num_attention_heads": 128,
    "num_kv_heads": 1,
    "d_c": 512,
    "d_r": 64,
    "mla": True,
}

MODEL_CONFIGS = {
    "llama4": config,
    "qwen3": config_1,
    "deepseek3": config_deepseek3,
}

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <model> <num_sa>")
        print(f"  model: {', '.join(MODEL_CONFIGS.keys())}")
        print(f"  num_sa: number of systolic arrays (integer)")
        sys.exit(1)

    model = sys.argv[1]
    if model not in MODEL_CONFIGS:
        print(f"Unknown model '{model}'. Choose from: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    try:
        num_sa = int(sys.argv[2])
    except ValueError:
        print(f"num_sa must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    cfg = MODEL_CONFIGS[model]

    bs_list = [1, 2, 4, 8, 16, 32, 64]
    seq_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    result = {}
     
    if not cfg.get("mla", False):
        R = cfg["num_q_head"] / cfg["num_kv_head"]

        n_qkv = cfg["num_q_head"] * cfg["d_head"] / 16 + 2 * cfg["num_kv_head"] * cfg["d_head"] / 16
        k_qkv = cfg["d_model"]
        proj_qkv = {}
        for bs in bs_list:
            overall, k_split = calculate_sa_qkv_utilization(bs, n_qkv, k_qkv, num_sa)
            proj_qkv[bs] = round(overall, 6)
        result["proj_qkv"] = proj_qkv

        score = {}
        for seq_len in seq_list:
            m = R
            n = seq_len / 4
            k = cfg["d_head"]
            overall, k_split = calculate_sa_score_utilization(m, n, k, num_sa)
            score[seq_len] = {"utilization": round(overall, 6), "k_split": k_split}
        result["score"] = score

        attention = {}
        for seq_len in seq_list:
            m = R
            k = seq_len / 4
            n = cfg["d_head"]
            overall, k_split = calculate_sa_att_utilization(m, n, k, num_sa)
            attention[seq_len] = {"utilization": round(overall, 6), "k_split": k_split}
        result["attention"] = attention

        proj_o = {}
        for bs in bs_list:
            m = bs
            n = cfg["d_model"]
            k = cfg["d_head"] * cfg["num_q_head"] / 16
            overall, k_split = calculate_sa_proj_o_utilization(m, n, k, num_sa)
            proj_o[bs] = round(overall, 6)
        result["proj_o"] = proj_o
    else:
        # DeepSeek MLA path
        d_cr = cfg["d_c"] + cfg["d_r"]
        n_heads = cfg["num_attention_heads"]
        n_kv = cfg["num_kv_heads"]

        n_qkv = n_heads * d_cr / 16 + n_kv * d_cr / 16
        k_qkv = cfg["d_model"]
        proj_qkv = {}
        for bs in bs_list:
            overall, k_split = calculate_sa_qkv_utilization(bs, n_qkv, k_qkv, num_sa)
            proj_qkv[bs] = round(overall, 6)
        result["proj_qkv"] = proj_qkv

        score = {}
        for seq_len in seq_list:
            m = n_heads
            n = seq_len / 16
            k = d_cr
            overall, k_split = calculate_sa_score_utilization(m, n, k, num_sa)
            score[seq_len] = {"utilization": round(overall, 6), "k_split": k_split}
        result["score"] = score

        attention = {}
        for seq_len in seq_list:
            m = n_heads
            n = cfg["d_c"]
            k = seq_len / 16
            overall, k_split = calculate_sa_att_utilization(m, n, k, num_sa)
            attention[seq_len] = {"utilization": round(overall, 6), "k_split": k_split}
        result["attention"] = attention

        proj_o = {}
        for bs in bs_list:
            m = bs
            n = cfg["d_model"]
            k = n_heads * cfg["d_c"] / 16
            overall, k_split = calculate_sa_proj_o_utilization(m, n, k, num_sa)
            proj_o[bs] = round(overall, 6)
        result["proj_o"] = proj_o

    out_file = f"{model}_util_{num_sa}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Generated {out_file}")

if __name__ == "__main__":
    main()