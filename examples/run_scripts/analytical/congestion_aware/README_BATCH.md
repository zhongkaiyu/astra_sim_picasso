# ASTRA-sim 批量执行脚本使用说明

## 快速开始

### 批量执行多个配置

```bash
# 从 configs/ 文件夹执行（只需文件名）
python3 batch_run_configs.py config1.json config2.json config3.json

# 或使用通配符执行 configs/ 文件夹中的所有 JSON 文件
python3 batch_run_configs.py configs/*.json
```

### 示例

```bash
cd examples/run_scripts/analytical/congestion_aware

# 执行两个配置进行对比（从 configs/ 文件夹读取）
python3 batch_run_configs.py \
    dp16_h100_mesh2d_4x4.json \
    dp16_h100_alltoall_16gpus.json

# 或使用完整路径
python3 batch_run_configs.py \
    configs/dp16_h100_mesh2d_4x4.json \
    configs/dp16_h100_alltoall_16gpus.json
```

### 目录结构

```
congestion_aware/
├── batch_run_configs.py          # 批量执行脚本
├── run_from_config.py            # 单个配置执行脚本
└── configs/                      # JSON 配置文件目录
    ├── dp16_h100_mesh2d_4x4.json
    ├── dp16_h100_alltoall_16gpus.json
    └── ...
```

## 输出结果

所有结果保存在 `output_qwen/batch_results_YYYYMMDD_HHMMSS/` 目录：

```
batch_results_20250101_120000/
├── batch_summary.txt          # 执行汇总报告
├── comparison_report.txt      # 性能对比报告（2个以上配置时）
├── logs/                      # 所有日志文件
└── analysis/                  # 详细分析报告
```

## 功能特性

- ✅ 自动批量执行多个 JSON 配置
- ✅ 自动收集和整理日志文件
- ✅ 自动生成性能分析报告
- ✅ 自动生成多配置对比报告
- ✅ 错误容错（单个失败不影响其他）
- ✅ 实时进度显示

## 查看结果

```bash
# 查看汇总报告
cat output_qwen/batch_results_*/batch_summary.txt

# 查看对比报告
cat output_qwen/batch_results_*/comparison_report.txt

# 查看某个配置的详细分析
cat output_qwen/batch_results_*/analysis/<config_name>_analysis.txt
```

## 注意事项

1. 确保 JSON 配置文件格式正确
2. 确保 workload 文件已存在
3. 确保 ASTRA-sim 已编译
4. 如果 `analyze_log.py` 不存在，会跳过详细分析但继续执行

## 相关文件

- `run_from_config.py` - 单个配置执行脚本
- `analyze_log.py` - 日志分析工具（项目根目录）

