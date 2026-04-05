import math
import json
import os

config = {
    "name": "llama3_70b",
    "d_model": 5120,
    "num_q_head": 40,
    "num_kv_head": 8,
    "d_head": 128
}

config_1 = {
    "name": "qwen3_235b",
    "d_model": 4096,
    "num_q_head": 64,
    "num_kv_head": 4,
    "d_head": 128
}

SA_SIZE = 16
NUM_SA = 96

def utilization_formula(k, sa, t=1):
    return (t * k) / (t * k + 2*sa - 1)

def weighted_k_util(k, sa, total_units, num_sa):
    '''Weighted average k_util across SAs with different tile counts.
    Some SAs get ceil(total/num_sa) tiles, others get floor(total/num_sa).'''
    max_tiles = math.ceil(total_units / num_sa)
    min_tiles = total_units // num_sa
    num_sa_with_max = total_units % num_sa
    num_sa_with_min = num_sa - num_sa_with_max

    if num_sa_with_max == 0:
        # All SAs have the same number of tiles
        return utilization_formula(k, sa, t=max_tiles)

    k_util_max = utilization_formula(k, sa, t=max_tiles)
    k_util_min = utilization_formula(k, sa, t=min_tiles) if min_tiles > 0 else 0
    return (num_sa_with_max * k_util_max + num_sa_with_min * k_util_min) / num_sa



def calculate_sa_qkv_utilization(m, n, k, num_sa=NUM_SA):
    '''
    Matmul: (m, k) @ (k, n)
    Distributed across num_sa (80) systolic arrays, each 16x16.

    Output has m_tiles * n_tiles total tiles, all distributed across 80 SAs.
    Tries k_split in [1, 2, 4] to improve SA distribution when output tiles are few.
    Utilization = tile_util * spatial_util * k_util
    '''
    sa = SA_SIZE

    m_tiles = math.ceil(m / sa)
    n_tiles = math.ceil(n / sa)
    total_tiles_mn = m_tiles * n_tiles

    # Spatial utilization (same for all k_splits)
    last_m_util = (m % sa / sa) if m % sa != 0 else 1.0
    last_n_util = (n % sa / sa) if n % sa != 0 else 1.0
    full_m = m_tiles - (1 if m % sa != 0 else 0)
    full_n = n_tiles - (1 if n % sa != 0 else 0)
    spatial_util = (
        full_m * full_n * 1.0 +
        full_m * (n_tiles - full_n) * last_n_util +
        (m_tiles - full_m) * full_n * last_m_util +
        (m_tiles - full_m) * (n_tiles - full_n) * last_m_util * last_n_util
    ) / total_tiles_mn

    best_sa_util = 0
    best_k_split = 1
    best_time = float('inf')

    for k_split in [1, 2, 3, 4]:
        k_per = k / k_split
        total_work_units = total_tiles_mn * k_split

        # Tile distribution across SAs
        max_units_per_sa = math.ceil(total_work_units / num_sa)
        tile_util = total_work_units / (max_units_per_sa * num_sa)

        # K utilization: weighted average across SAs with different tile counts
        k_util = weighted_k_util(k_per, sa, total_work_units, num_sa)

        sa_util = tile_util * spatial_util * k_util

        # Decision: wall-clock determined by slowest SA (max tiles)
        sa_cycles = max_units_per_sa * k_per + 2 * sa - 1
        num_reductions = reduction_cost(k_split, m, n)
        total_time = sa_cycles + num_reductions
        if total_time < best_time:
            best_time = total_time
            best_sa_util = sa_util
            best_k_split = k_split

    return best_sa_util, best_k_split


def reduction_cost(k_split, m, n, sa=SA_SIZE):
    '''Returns the reduction cost on vector unit for a given k_split.
    Tree reduction: k_split partials per output tile -> ceil(log2(k_split)) stages.
    Each output tile needs its own independent reduction.
    Each stage costs 2*sa - 1 = 31 cycles.
    Total = output_tiles * ceil(log2(k_split)) * (2*sa - 1)
    '''
    if k_split <= 1:
        return 0
    m_tiles = math.ceil(m / sa)
    n_tiles = math.ceil(n / sa)
    output_tiles = m_tiles * n_tiles
    cycles_per_stage = (2 * sa - 1) * 2
    return math.ceil(math.log2(k_split)) * cycles_per_stage

def calculate_sa_score_utilization(m, n, k, num_sa=NUM_SA):
    '''
    Matmul: (m, k) @ (k, n) for score (QK^T).
    k is fixed (d_head=128), m and n vary.
    When m*n is small, splits k to distribute more work across SAs.
    Uses double buffering for k utilization.
    Returns pure SA utilization (reduction cost not included).
    '''
    sa = SA_SIZE

    m_tiles = math.ceil(m / sa)
    n_tiles = math.ceil(n / sa)

    # Spatial utilization (same as Q projection)
    last_m_util = (m % sa / sa) if m % sa != 0 else 1.0
    last_n_util = (n % sa / sa) if n % sa != 0 else 1.0
    full_m = m_tiles - (1 if m % sa != 0 else 0)
    full_n = n_tiles - (1 if n % sa != 0 else 0)
    total_tiles_mn = m_tiles * n_tiles
    spatial_util = (
        full_m * full_n * 1.0 +
        full_m * (n_tiles - full_n) * last_n_util +
        (m_tiles - full_m) * full_n * last_m_util +
        (m_tiles - full_m) * (n_tiles - full_n) * last_m_util * last_n_util
    ) / total_tiles_mn

    best_sa_util = 0
    best_k_split = 1
    best_time = float('inf')

    for k_split in [1, 2, 4, 6, 8, 10, 12]:
        k_per = k / k_split
        total_work_units = total_tiles_mn * k_split

        # Tile distribution across SAs
        max_units_per_sa = math.ceil(total_work_units / num_sa)
        tile_util = total_work_units / (max_units_per_sa * num_sa)

        # K utilization: single buffering, amortized over t tiles
        k_util = utilization_formula(k_per, sa, t=max_units_per_sa)

        sa_util = tile_util * spatial_util * k_util

        # Decision: compare total wall-clock time (SA cycles + reduction on vector unit)
        sa_cycles = max_units_per_sa * k_per + 2 * sa - 1
        num_reductions = reduction_cost(k_split, m, n)
        total_time = sa_cycles + num_reductions

        if total_time < best_time:
            best_time = total_time
            best_sa_util = sa_util
            best_k_split = k_split


    return best_sa_util, best_k_split

def calculate_sa_att_utilization(m, n, k, num_sa=NUM_SA):
    '''
    Matmul: (m, k) @ (k, n) for attention (Score @ V).
    n is fixed (d_head=128), m and k vary.
    Compares OS (no split) vs OS + k-split.
    Uses double buffering for k utilization.
    '''
    sa = SA_SIZE

    m_tiles = math.ceil(m / sa)
    n_tiles = math.ceil(n / sa)
    total_tiles_mn = m_tiles * n_tiles

    # Spatial utilization (same for all splits)
    last_m_util = (m % sa / sa) if m % sa != 0 else 1.0
    last_n_util = (n % sa / sa) if n % sa != 0 else 1.0
    full_m = m_tiles - (1 if m % sa != 0 else 0)
    full_n = n_tiles - (1 if n % sa != 0 else 0)
    spatial_util = (
        full_m * full_n * 1.0 +
        full_m * (n_tiles - full_n) * last_n_util +
        (m_tiles - full_m) * full_n * last_m_util +
        (m_tiles - full_m) * (n_tiles - full_n) * last_m_util * last_n_util
    ) / total_tiles_mn

    # Try k-split factors: 1 (no split), 2, 4, 8
    best_sa_util = 0
    best_k_split = 1
    best_time = float('inf')

    for k_split in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60]:
        k_per = k / k_split
        if k_per < 1:
            continue
        total_work_units = total_tiles_mn * k_split

        # Tile distribution across SAs
        max_units_per_sa = math.ceil(total_work_units / num_sa)
        tile_util = total_work_units / (max_units_per_sa * num_sa)

        # K utilization: single buffering, amortized over t tiles
        k_util = utilization_formula(k_per, sa, t=max_units_per_sa)

        sa_util = tile_util * spatial_util * k_util

        # Decision: compare total wall-clock time
        sa_cycles = max_units_per_sa * k_per + 2 * sa - 1
        num_reductions = reduction_cost(k_split, m, n)
        total_time = sa_cycles + num_reductions

        if total_time < best_time:
            best_time = total_time
            best_sa_util = sa_util
            best_k_split = k_split

    return best_sa_util, best_k_split

def calculate_sa_proj_o_utilization(m, n, k, num_sa=NUM_SA):
    return calculate_sa_qkv_utilization(m, n, k, num_sa)


if __name__ == "__main__":
    n = config["num_q_head"] * config["d_head"] / 16 + 2 * config["num_kv_head"] * config["d_head"] / 16
    k = config["d_model"]

    print("projection QKV utilization: ")
    print()
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        overall, k_sp = calculate_sa_qkv_utilization(bs, n, k)
        print(f"bs={bs:>4d} utilization={overall:.4f} k_split={k_sp}")

    print()
    

    print("score (QK^T) utilization: ")
    print()
    R = config["num_q_head"] / config["num_kv_head"]
    for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        m = R
        n = seq_len / 4
        k = config["d_head"]
        overall = calculate_sa_score_utilization(m, n, k)
        print(f"seq_len={seq_len:>7d} utilization={overall[0]:.4f} k_split={overall[1]}")
    print()

    print("attention utilization: ")
    print()
    for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        m = R
        k = seq_len / 4
        n = config["d_head"]
        overall, k_sp = calculate_sa_att_utilization(m, n, k)
        print(f"seq_len={seq_len:>7d} utilization={overall:.4f} k_split={k_sp}")
    print()


    print("projection O utilization: ")
    print()
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        m = bs
        n = config["d_model"]
        k = config["d_head"] * config["num_q_head"] / 16
        overall, k_sp = calculate_sa_proj_o_utilization(m, n, k)
        print(f"bs={bs:>4d} utilization={overall:.4f} k_split={k_sp}")

    # --- Save utilization data to JSON ---
    result = {}

    # proj_qkv
    n_qkv = config["num_q_head"] * config["d_head"] / 16 + 2 * config["num_kv_head"] * config["d_head"] / 16
    k_qkv = config["d_model"]
    proj_qkv_data = {}
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        overall, _ = calculate_sa_qkv_utilization(bs, n_qkv, k_qkv)
        proj_qkv_data[str(bs)] = round(overall, 6)
    result["proj_qkv"] = proj_qkv_data

    # score (QK^T)
    R = config["num_q_head"] / config["num_kv_head"]
    score_data = {}
    for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        m = R
        n = seq_len / 4
        k = config["d_head"]
        overall, k_sp = calculate_sa_score_utilization(m, n, k)
        score_data[str(seq_len)] = {"utilization": round(overall, 6), "k_split": k_sp}
    result["score"] = score_data

    # attention (Score @ V)
    att_data = {}
    for seq_len in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]:
        m = R
        k = seq_len / 4
        n = config["d_head"]
        overall, k_sp = calculate_sa_att_utilization(m, n, k)
        att_data[str(seq_len)] = {"utilization": round(overall, 6), "k_split": k_sp}
    result["attention"] = att_data

    # proj_o
    proj_o_data = {}
    for bs in [1, 2, 4, 8, 16, 32, 64]:
        m = bs
        n = config["d_model"]
        k = config["d_head"] * config["num_q_head"] / 16
        overall, _ = calculate_sa_proj_o_utilization(m, n, k)
        proj_o_data[str(bs)] = round(overall, 6)
    result["proj_o"] = proj_o_data

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = config["name"]
    output_path = os.path.join(script_dir, f"{model_name}_utilization_{NUM_SA}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved utilization data to {output_path}")

