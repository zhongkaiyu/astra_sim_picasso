B200_config = {
    "compute": 9000,
    "Bandwidth": 8, # TB/s
    "capacity": 192, # GB
    "power": 1000, # W
    "kv_cache_size": None, # GB, set via set_kv_cache_size()
    "num_devices": 2,
    "device_link_bw": 1.8 # TB/s
}
B200_config["kv_cache_size"] = B200_config["capacity"] * B200_config["num_devices"] - 235 - 5

ours_config = {
    "compute": 640,
    "Bandwidth": 44, # TB/s
    "capacity": 576, # GB
    "power": 1244, # W
    "kv_cache_size": None, # GB
    "num_devices": 1,
    "device_link_bw": 1.8 # TB/s
}
ours_config["kv_cache_size"] = ours_config["capacity"] * ours_config["num_devices"] - 235 - 5

rubin_config = {
    "compute": 17500, # TFLOPs
    "Bandwidth": 22, # TB/s
    "capacity": 288, # GB
    "power": 2200, # W
    "kv_cache_size": None, # GB
    "num_devices": 1,
    "device_link_bw": 1.8 # TB/s
}
rubin_config["kv_cache_size"] = rubin_config["capacity"] * rubin_config["num_devices"] - 235 - 5

# ---------------------------------------------------------------------------
# Per-NPU configs for AstraSim GQA single-layer analysis
#   - Used with strategy-aware model configs in attention.py
#   - AstraSim roofline: peak-perf (TFLOPS) and local-mem-bw (GB/s)
#   - Trace encodes: num_ops = MAC count; C++ converts to FLOPs (2鑴矼ACs)
#   - So "compute" = peak-perf TFLOPS, "Bandwidth" = local-mem-bw in TB/s
# ---------------------------------------------------------------------------

hbm4_npu_config = {
    "compute": 40,          # TFLOPS = peak-perf from onering16_HBM4.json
    "Bandwidth": 2.5,       # TB/s (= T-elements/s at FP8, local-mem-bw=2500)
    "capacity": 36,         # GB per NPU (576/16)
    "power": 78,            # W per NPU (1244/16)
    "kv_cache_size": None,
    "num_devices": 1,
    "device_link_bw": 2.0,  # TB/s D2D
    "kv_bytes_per_element": 1,  # element count = byte count at FP8
}

H100_fp8_config = {
    "compute": 1979,        # TFLOPS (H100 SXM FP8 dense Tensor Core)
    "Bandwidth": 3.35,      # TB/s (HBM3)
    "capacity": 80,         # GB per GPU
    "power": 700,           # W
    "kv_cache_size": None,  # GB (set below, assumes 8-way TP for Qwen3 235B)
    "num_devices": 1,       # per-GPU roofline (1 GPU)
    "device_link_bw": 0.9,  # TB/s NVLink per direction (900 GB/s)
    "kv_bytes_per_element": 1,
}
# 8-way TP: each GPU holds 235/8 ≈ 29.4 GB weights, rest for KV cache
H100_fp8_config["kv_cache_size"] = H100_fp8_config["capacity"] - 235 / 8 - 5

rubin_single_layer_config = {
    "compute": 17500,       # TFLOPS (Rubin spec)
    "Bandwidth": 22,        # TB/s (= T-elements/s at FP8)
    "capacity": 288,
    "power": 2200,
    "kv_cache_size": None,
    "num_devices": 1,
    "device_link_bw": 1.8,
    "kv_bytes_per_element": 1,
}