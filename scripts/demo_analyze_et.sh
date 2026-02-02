#!/bin/bash
# ET 文件分析演示脚本
# 展示如何使用 analyze_et_file.py 工具

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ET 文件分析工具演示${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# 查找示例 ET 文件
EXAMPLE_ET=""
if [ -f "$PROJECT_ROOT/output/megatron_tp8/workloads/attention_tp8.0.et" ]; then
    EXAMPLE_ET="$PROJECT_ROOT/output/megatron_tp8/workloads/attention_tp8.0.et"
elif [ -f "$PROJECT_ROOT/symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8.0.et" ]; then
    EXAMPLE_ET="$PROJECT_ROOT/symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8.0.et"
else
    echo -e "${YELLOW}??  未找到示例 ET 文件${NC}"
    echo ""
    echo "请先生成 workload:"
    echo "  cd symbolic_tensor_graph_picasso"
    echo "  python main.py --output_dir generated_attn_8npu/ --output_name attention_tp8 \\"
    echo "      --model_type moe_attention --tp 8 --num_stacks 1 --dmodel 4096 \\"
    echo "      --head 64 --kvhead 4 --batch 1 --seq 1024"
    echo ""
    exit 1
fi

echo "找到示例文件: $EXAMPLE_ET"
echo ""

# 演示 1: 基本分析
echo -e "${GREEN}演示 1: 基本分析${NC}"
echo "命令: python analyze_et_file.py --input $EXAMPLE_ET"
echo ""
python3 "$SCRIPT_DIR/analyze_et_file.py" --input "$EXAMPLE_ET" | head -40
echo ""
echo -e "${YELLOW}(输出已截断，显示前 40 行)${NC}"
echo ""
read -p "按 Enter 继续..."
echo ""

# 演示 2: 保存报告到文件
echo -e "${GREEN}演示 2: 保存报告到文件${NC}"
REPORT_FILE="/tmp/et_analysis_report.txt"
echo "命令: python analyze_et_file.py --input $EXAMPLE_ET --output $REPORT_FILE"
echo ""
python3 "$SCRIPT_DIR/analyze_et_file.py" --input "$EXAMPLE_ET" --output "$REPORT_FILE"
echo ""
echo "报告已保存到: $REPORT_FILE"
echo "查看文件内容:"
head -30 "$REPORT_FILE"
echo ""
echo -e "${YELLOW}(显示前 30 行)${NC}"
echo ""
read -p "按 Enter 继续..."
echo ""

# 演示 3: 详细统计信息
echo -e "${GREEN}演示 3: 显示详细统计信息${NC}"
echo "命令: python analyze_et_file.py --input $EXAMPLE_ET --stats"
echo ""
python3 "$SCRIPT_DIR/analyze_et_file.py" --input "$EXAMPLE_ET" --stats 2>/dev/null | tail -30
echo ""
echo -e "${YELLOW}(显示输出的最后 30 行)${NC}"
echo ""
read -p "按 Enter 继续..."
echo ""

# 演示 4: 导出图结构
echo -e "${GREEN}演示 4: 导出图结构${NC}"
GRAPH_FILE="/tmp/et_graph_structure.txt"
echo "命令: python analyze_et_file.py --input $EXAMPLE_ET --graph $GRAPH_FILE"
echo ""
python3 "$SCRIPT_DIR/analyze_et_file.py" --input "$EXAMPLE_ET" --graph "$GRAPH_FILE" 2>/dev/null
echo ""
echo "图结构已保存到: $GRAPH_FILE"
echo "文件前几行:"
head -20 "$GRAPH_FILE"
echo ""
echo -e "${YELLOW}(显示前 20 行)${NC}"
echo ""
read -p "按 Enter 继续..."
echo ""

# 演示 5: 批量分析
echo -e "${GREEN}演示 5: 批量分析（分析所有 NPU 的 workload）${NC}"
echo ""
if [ -d "$PROJECT_ROOT/output/megatron_tp8/workloads" ]; then
    echo "命令: bash batch_analyze_et.sh -i ../output/megatron_tp8/workloads -p attention_tp8 -n 8"
    echo ""
    bash "$SCRIPT_DIR/batch_analyze_et.sh" \
        -i "$PROJECT_ROOT/output/megatron_tp8/workloads" \
        -p "attention_tp8" \
        -n 8 \
        -o "/tmp/batch_analysis"
    echo ""
    echo "批量分析结果已保存到: /tmp/batch_analysis/"
    echo ""
    if [ -f "/tmp/batch_analysis/comparison.csv" ]; then
        echo "对比数据:"
        cat "/tmp/batch_analysis/comparison.csv"
    fi
elif [ -d "$PROJECT_ROOT/symbolic_tensor_graph_picasso/generated_attn_8npu" ]; then
    echo "命令: bash batch_analyze_et.sh -i ../symbolic_tensor_graph_picasso/generated_attn_8npu -p attention_tp8 -n 8"
    echo ""
    bash "$SCRIPT_DIR/batch_analyze_et.sh" \
        -i "$PROJECT_ROOT/symbolic_tensor_graph_picasso/generated_attn_8npu" \
        -p "attention_tp8" \
        -n 8 \
        -o "/tmp/batch_analysis"
    echo ""
    echo "批量分析结果已保存到: /tmp/batch_analysis/"
    echo ""
    if [ -f "/tmp/batch_analysis/comparison.csv" ]; then
        echo "对比数据:"
        cat "/tmp/batch_analysis/comparison.csv"
    fi
else
    echo -e "${YELLOW}??  未找到批量分析的目录${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}? 演示完成！${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "更多使用方法，请参考:"
echo "  cat $SCRIPT_DIR/README_ANALYZE_ET.md"
echo ""
echo "查看帮助:"
echo "  python $SCRIPT_DIR/analyze_et_file.py --help"
echo "  bash $SCRIPT_DIR/batch_analyze_et.sh --help"
echo ""

