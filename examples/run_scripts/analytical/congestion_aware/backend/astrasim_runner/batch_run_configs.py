#!/usr/bin/env python3
# ============================================================================
# ASTRA-sim 批量执行脚本
# ============================================================================
# 功能：批量执行多个 JSON 配置文件，收集日志并生成分析报告
#
# 用法：
#   python3 batch_run_configs.py config1.json config2.json config3.json ...
#   或者：
#   python3 batch_run_configs.py configs/*.json
#
# 说明：
#   - JSON 配置文件默认从 configs/ 文件夹读取
#   - 支持相对路径和绝对路径
#   - 如果只提供文件名，会自动在 configs/ 文件夹中查找
#
# 输出：
#   - 每个配置的日志文件保存在各自的 output_dir/logs/ 目录
#   - 汇总分析报告保存在 batch_results/ 目录
# ============================================================================

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[{}] 0.0%".format(" " * width)
    ratio = max(0.0, min(1.0, current / total))
    filled = int(round(ratio * width))
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


def resolve_config_path(config_path: str, configs_dir: Path) -> str:
    if config_path.startswith("/") and Path(config_path).is_file():
        return config_path

    path = Path(config_path)
    if path.is_file():
        return str(path.resolve())

    candidate = configs_dir / config_path
    if candidate.is_file():
        return str(candidate)

    basename_candidate = configs_dir / Path(config_path).name
    if basename_candidate.is_file():
        return str(basename_candidate)

    return config_path


def format_config_value(value, project_dir: Path, example_dir: Path):
    if isinstance(value, str):
        return value.format(PROJECT_DIR=str(project_dir), EXAMPLE_DIR=str(example_dir))
    return value


def read_last_lines(path: Path, count: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-count:]
    except FileNotFoundError:
        return []


def extract_metrics(log_path: Path) -> dict[str, str]:
    metrics = {}
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return metrics

    wall_match = re.search(r"sys\[0\].*Wall time: (\d+)", content)
    gpu_match = re.search(r"sys\[0\].*GPU time: (\d+)", content)
    comm_match = re.search(r"sys\[0\].*Comm time: (\d+)", content)

    if wall_match:
        metrics["Wall Time"] = wall_match.group(1)
    if gpu_match:
        metrics["GPU Time"] = gpu_match.group(1)
    if comm_match:
        metrics["Comm Time"] = comm_match.group(1)
    return metrics


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_dir = (script_dir / "../../../..").resolve()
    configs_dir = script_dir / "configs"
    run_script = script_dir / "run_from_config.py"
    analyze_script = project_dir / "analyze_log.py"

    if not run_script.is_file():
        print(f"ERROR: Run script not found: {run_script}")
        return 1

    if not analyze_script.is_file():
        print(f"WARNING: Analyze script not found: {analyze_script}")
        print("Will skip detailed analysis, only collect logs.")
        analyze_script = None

    if len(sys.argv) == 1:
        print("用法: python3 batch_run_configs.py <config1.json> [config2.json] ...")
        print("")
        print("示例:")
        print("  # 从 configs/ 文件夹读取（只需文件名）")
        print(
            "  python3 batch_run_configs.py dp16_h100_mesh2d_4x4.json "
            "dp16_h100_alltoall_16gpus.json"
        )
        print("  python3 batch_run_configs.py configs/*.json")
        print("")
        print("  # 使用完整路径")
        print("  python3 batch_run_configs.py configs/dp16_h100_mesh2d_4x4.json")
        return 1

    batch_results_dir = project_dir / f"output_qwen/batch_results_{datetime.now():%Y%m%d_%H%M%S}"
    (batch_results_dir / "logs").mkdir(parents=True, exist_ok=True)
    (batch_results_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (batch_results_dir / "summaries").mkdir(parents=True, exist_ok=True)

    config_names: list[str] = []
    log_files: list[str] = []
    output_dirs: list[str] = []
    status: list[str] = []

    total_configs = len(sys.argv) - 1
    success_count = 0
    fail_count = 0
    current = 0

    print("==========================================================================")
    print("ASTRA-sim 批量执行脚本")
    print("==========================================================================")
    print(f"配置数量: {total_configs}")
    print(f"结果目录: {batch_results_dir}")
    print("==========================================================================")
    print("")

    for config_arg in sys.argv[1:]:
        current += 1
        config_json = resolve_config_path(config_arg, configs_dir)

        progress = render_progress(current, total_configs)
        print(f"进度: {progress}")

        if not Path(config_json).is_file():
            print(f"[{current}/{total_configs}] ❌ 配置文件不存在: {config_arg}")
            print(f"  尝试路径: {config_json}")
            config_name = Path(config_arg).stem
            config_names.append(config_name)
            log_files.append("")
            output_dirs.append("")
            status.append("FAILED: File not found")
            fail_count += 1
            continue

        config_name = Path(config_json).stem
        config_names.append(config_name)

        print(f"[{current}/{total_configs}] 正在执行: {config_name}")
        print(f"  配置文件: {config_json}")
        if config_json != config_arg:
            print(f"  (从 {config_arg} 解析)")

        run_log_path = batch_results_dir / "logs" / f"{config_name}.run.log"
        with run_log_path.open("w", encoding="utf-8") as run_log:
            result = subprocess.run(
                ["python3", str(run_script), config_json],
                stdout=run_log,
                stderr=run_log,
                check=False,
            )

        if result.returncode == 0:
            print("  ✓ 执行成功")
            success_count += 1

            try:
                with Path(config_json).open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                cfg = {}

            example_dir = project_dir / "examples"
            output_dir = format_config_value(cfg.get("output_dir", ""), project_dir, example_dir)
            log_file = cfg.get("log_file", "")

            output_dir_path = str(output_dir) if output_dir else ""
            log_file_name = str(log_file) if log_file else ""

            if output_dir_path and log_file_name:
                log_file_path = Path(output_dir_path) / "logs" / log_file_name
                output_dirs.append(output_dir_path)

                if log_file_path.is_file():
                    batch_log_path = batch_results_dir / "logs" / f"{config_name}.log"
                    shutil.copy2(log_file_path, batch_log_path)
                    log_files.append(str(batch_log_path))
                    status.append("SUCCESS")

                    print(f"  日志文件: {log_file_path}")

                    if analyze_script:
                        print("  正在分析日志...")
                        analysis_path = batch_results_dir / "analysis" / f"{config_name}_analysis.txt"
                        with analysis_path.open("w", encoding="utf-8") as analysis_out:
                            analysis_result = subprocess.run(
                                [
                                    "python3",
                                    str(analyze_script),
                                    str(log_file_path),
                                    "--output",
                                    str(analysis_path),
                                    "--no-save",
                                ],
                                stdout=analysis_out,
                                stderr=subprocess.STDOUT,
                                check=False,
                            )
                        if analysis_result.returncode == 0:
                            print("  ✓ 分析完成")
                        else:
                            print("  ⚠ 分析失败（继续执行）")
                else:
                    log_files.append("")
                    status.append("WARNING: Log file not found")
                    print(f"  ⚠ 日志文件未找到: {log_file_path}")
            else:
                log_files.append("")
                output_dirs.append(output_dir_path)
                status.append("WARNING: Could not extract output paths")
                print("  ⚠ 无法提取输出路径")
        else:
            print("  ❌ 执行失败")
            fail_count += 1
            log_files.append("")
            output_dirs.append("")
            status.append("FAILED: Execution error")

            if run_log_path.is_file():
                print("  错误信息:")
                for line in read_last_lines(run_log_path, 5):
                    print(f"    {line.rstrip()}")
        print("")

    print("==========================================================================")
    print("生成汇总报告...")
    print("==========================================================================")

    summary_file = batch_results_dir / "batch_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("ASTRA-sim 批量执行汇总报告\n")
        f.write(f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("==========================================================================\n")
        f.write("\n")
        f.write("执行统计:\n")
        f.write(f"  总配置数: {total_configs}\n")
        f.write(f"  成功: {success_count}\n")
        f.write(f"  失败: {fail_count}\n")
        f.write("\n")
        f.write("==========================================================================\n")
        f.write("详细结果:\n")
        f.write("==========================================================================\n")
        f.write("\n")

        for i, name in enumerate(config_names):
            f.write(f"[{i + 1}] {name}\n")
            f.write(f"    状态: {status[i]}\n")
            if output_dirs[i]:
                f.write(f"    输出目录: {output_dirs[i]}\n")
            if log_files[i] and Path(log_files[i]).is_file():
                f.write(f"    日志文件: {log_files[i]}\n")

                metrics = extract_metrics(Path(log_files[i]))
                if metrics.get("Wall Time"):
                    f.write(f"    Wall Time: {metrics['Wall Time']} cycles\n")
                if metrics.get("GPU Time"):
                    f.write(f"    GPU Time: {metrics['GPU Time']} cycles\n")
                if metrics.get("Comm Time"):
                    f.write(f"    Comm Time: {metrics['Comm Time']} cycles\n")
            f.write("\n")

    valid_logs = []
    valid_names = []
    for i, log_path in enumerate(log_files):
        if log_path and Path(log_path).is_file() and status[i] == "SUCCESS":
            valid_logs.append(log_path)
            valid_names.append(config_names[i])

    if len(valid_logs) >= 2 and analyze_script:
        print("生成对比分析报告...")
        compare_file = batch_results_dir / "comparison_report.txt"
        subprocess.run(
            ["python3", str(analyze_script), *valid_logs, "--compare", "--output", str(compare_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if compare_file.is_file():
            print(f"  ✓ 对比报告: {compare_file}")

    print("")
    print("==========================================================================")
    print("批量执行完成！")
    print("==========================================================================")
    print(f"结果目录: {batch_results_dir}")
    print("")
    print("文件结构:")
    print(f"  {batch_results_dir}/")
    print("    ├── batch_summary.txt          # 汇总报告")
    if (batch_results_dir / "comparison_report.txt").is_file():
        print("    ├── comparison_report.txt      # 对比分析报告")
    print("    ├── logs/                      # 所有日志文件")
    print("    │   ├── <config1>.log")
    print("    │   ├── <config2>.log")
    print("    │   └── ...")
    print("    └── analysis/                  # 详细分析报告")
    print("        ├── <config1>_analysis.txt")
    print("        └── ...")
    print("")
    print("查看汇总报告:")
    print(f"  cat {summary_file}")
    print("")
    if (batch_results_dir / "comparison_report.txt").is_file():
        print("查看对比报告:")
        print(f"  cat {batch_results_dir / 'comparison_report.txt'}")
        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())

