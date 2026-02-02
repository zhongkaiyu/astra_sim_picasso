#!/bin/bash
# 批量分析 ET 文件脚本
# 用途：分析一组 ET 文件并生成对比报告

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_et_file.py"

# 默认值
INPUT_DIR=""
OUTPUT_DIR="$SCRIPT_DIR/analysis_results"
PREFIX=""
NUM_FILES=8

# 打印帮助
print_help() {
    cat << EOF
批量分析 Chakra ET 文件

用法:
    $0 [选项]

选项:
    -i, --input DIR       输入目录（包含 .et 文件）
    -o, --output DIR      输出目录（默认: ./analysis_results）
    -p, --prefix PREFIX   文件前缀（例如: attention_tp8）
    -n, --num NUM         文件数量（默认: 8）
    -h, --help            显示帮助信息

示例:
    # 分析 Megatron TP8 workload
    $0 -i ../output/megatron_tp8/workloads -p attention_tp8 -n 8

    # 分析 Mesh2D TP24 workload
    $0 -i ../symbolic_tensor_graph_picasso/generated_attn_24npu -p attention_tp24 -n 24

    # 自定义输出目录
    $0 -i ../output/megatron_tp8/workloads -p attention_tp8 -o my_results

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        -n|--num)
            NUM_FILES="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$INPUT_DIR" ] || [ -z "$PREFIX" ]; then
    echo -e "${YELLOW}错误: 必须指定输入目录和文件前缀${NC}"
    print_help
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  批量分析 Chakra ET 文件${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "配置信息:"
echo "  输入目录:   $INPUT_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  文件前缀:   $PREFIX"
echo "  文件数量:   $NUM_FILES"
echo ""

# 初始化汇总文件
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
COMPARISON_FILE="$OUTPUT_DIR/comparison.csv"

echo "# ET 文件分析汇总报告" > "$SUMMARY_FILE"
echo "# 生成时间: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

echo "NPU,Total_Nodes,COMP_Nodes,COMM_Nodes,Total_Ops_GFLOPS,Total_Comm_Size_MB" > "$COMPARISON_FILE"

# 分析每个文件
for i in $(seq 0 $((NUM_FILES - 1))); do
    INPUT_FILE="$INPUT_DIR/${PREFIX}.${i}.et"
    OUTPUT_FILE="$OUTPUT_DIR/report_npu${i}.txt"
    
    echo -e "${BLUE}[$(($i + 1))/$NUM_FILES]${NC} 分析 NPU $i: $INPUT_FILE"
    
    # 检查文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "  ${YELLOW}??  文件不存在，跳过${NC}"
        continue
    fi
    
    # 运行分析
    if python3 "$ANALYZE_SCRIPT" --input "$INPUT_FILE" --output "$OUTPUT_FILE" > /dev/null 2>&1; then
        echo -e "  ${GREEN}? 完成${NC}"
        
        # 提取关键指标到汇总文件
        echo "=== NPU $i ===" >> "$SUMMARY_FILE"
        grep -E "(Total Nodes|COMP|COMM_COLL|Total Operations|Total Communication Size)" "$OUTPUT_FILE" >> "$SUMMARY_FILE" 2>/dev/null || true
        echo "" >> "$SUMMARY_FILE"
        
        # 提取数据到 CSV
        TOTAL_NODES=$(grep "Total Nodes:" "$OUTPUT_FILE" | awk '{print $3}' | tr -d ',' || echo "0")
        COMP_NODES=$(grep "^COMP " "$OUTPUT_FILE" | awk '{print $2}' | tr -d ',' || echo "0")
        COMM_NODES=$(grep "^COMM_COLL " "$OUTPUT_FILE" | awk '{print $2}' | tr -d ',' || echo "0")
        
        # 提取操作数和通信大小
        TOTAL_OPS=$(grep "Total Operations:" "$OUTPUT_FILE" | awk '{print $3}' || echo "0")
        COMM_SIZE=$(grep "Total Communication Size:" "$OUTPUT_FILE" | awk '{print $4}' || echo "0")
        
        echo "$i,$TOTAL_NODES,$COMP_NODES,$COMM_NODES,$TOTAL_OPS,$COMM_SIZE" >> "$COMPARISON_FILE"
    else
        echo -e "  ${YELLOW}? 分析失败${NC}"
    fi
done

# 生成对比分析
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  对比分析${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# 打印表格
if command -v column &> /dev/null; then
    echo "NPU 节点统计对比:"
    cat "$COMPARISON_FILE" | column -t -s','
else
    echo "NPU 节点统计对比 (安装 'column' 命令可获得更好的显示):"
    cat "$COMPARISON_FILE"
fi

echo ""
echo -e "${GREEN}? 分析完成！${NC}"
echo ""
echo "输出文件:"
echo "  汇总报告:   $SUMMARY_FILE"
echo "  对比数据:   $COMPARISON_FILE"
echo "  详细报告:   $OUTPUT_DIR/report_npu*.txt"
echo ""
echo "查看结果:"
echo "  cat $SUMMARY_FILE"
echo "  cat $COMPARISON_FILE"
echo ""

