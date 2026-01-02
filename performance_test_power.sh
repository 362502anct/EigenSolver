#!/bin/bash
# 性能测试脚本 - 使用cases目录中的测试矩阵和幂法计算最大特征值

echo "========================================"
echo "EigenSolver 幂法性能测试"
echo "测试cases目录中的真实矩阵"
echo "========================================"
echo ""

# 检查可执行文件是否存在
if [ ! -f "./eigensolver" ]; then
    echo "错误: 找不到可执行文件 ./eigensolver"
    echo "请先运行: mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 检查cases目录是否存在
if [ ! -d "./cases" ]; then
    echo "错误: 找不到cases目录"
    exit 1
fi

# 定义颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取cases目录中的所有.mtx和.mtx.gz文件
CASES_DIR="./cases"
MATRIX_FILES=$(find "$CASES_DIR" -type f \( -name "*.mtx" -o -name "*.mtx.gz" \) | sort)

if [ -z "$MATRIX_FILES" ]; then
    echo "警告: 在 $CASES_DIR 目录中未找到任何.mtx或.mtx.gz文件"
    echo ""
    echo "测试随机对称矩阵作为替代..."
    echo ""
else
    echo -e "${BLUE}找到以下测试矩阵:${NC}"
    echo "$MATRIX_FILES" | while read -r file; do
        echo "  - $(basename "$file")"
    done
    echo ""
fi

# 创建结果目录
RESULT_DIR="./performance_results"
mkdir -p "$RESULT_DIR"

# 结果文件
RESULT_FILE="$RESULT_DIR/power_method_results_$(date +%Y%m%d_%H%M%S).txt"
SUMMARY_FILE="$RESULT_DIR/summary.txt"

echo "========================================"
echo -e "${GREEN}开始性能测试${NC}"
echo "========================================"
echo ""

# 输出文件头
{
    echo "========================================"
    echo "EigenSolver 幂法性能测试结果"
    echo "测试时间: $(date)"
    echo "========================================"
    echo ""
} | tee "$RESULT_FILE"

# 测试配置
TOLERANCE="1e-10"
MAX_ITERATIONS="10000"

# 如果没有找到矩阵文件，测试随机矩阵
if [ -z "$MATRIX_FILES" ]; then
    echo -e "${YELLOW}测试随机对称矩阵${NC}"
    echo "========================================"

    for size in 100 500 1000 2000; do
        echo ""
        echo -e "${BLUE}矩阵大小: ${size}x${size}${NC}"
        echo "----------------------------------------"

        {
            echo "测试: 随机对称矩阵 ${size}x${size}"
            echo "方法: Power Method"
            echo "容差: $TOLERANCE"
            echo "最大迭代次数: $MAX_ITERATIONS"
            echo ""
        } | tee -a "$RESULT_FILE"

        # 运行测试
        /usr/bin/time -f "执行时间: %E (用户: %U, 系统: %S, CPU%: %P)" \
            ./eigensolver -s $size -m power -t $TOLERANCE -i $MAX_ITERATIONS \
            2>&1 | tee -a "$RESULT_FILE"

        echo "" | tee -a "$RESULT_FILE"
    done
else
    # 测试cases目录中的矩阵文件
    echo "$MATRIX_FILES" | while read -r matrix_file; do
        if [ -f "$matrix_file" ]; then
            echo ""
            echo -e "${BLUE}测试矩阵: $(basename "$matrix_file")${NC}"
            echo "========================================"

            {
                echo ""
                echo "========================================"
                echo "测试矩阵: $matrix_file"
                echo "方法: Power Method (计算最大特征值)"
                echo "容差: $TOLERANCE"
                echo "最大迭代次数: $MAX_ITERATIONS"
                echo "========================================"
                echo ""
            } | tee -a "$RESULT_FILE"

            # 运行测试并捕获输出
            /usr/bin/time -f "执行时间: %E (用户: %U, 系统: %S, CPU%: %P)" \
                ./eigensolver -f "$matrix_file" -m power -t $TOLERANCE -i $MAX_ITERATIONS \
                2>&1 | tee -a "$RESULT_FILE"

            echo "" | tee -a "$RESULT_FILE"
        fi
    done
fi

# 生成摘要
{
    echo ""
    echo "========================================"
    echo "测试摘要"
    echo "========================================"
    echo "结果保存在: $RESULT_FILE"
    echo ""
    echo "对比不同方法:"
    echo "----------------------------------------"
} | tee -a "$RESULT_FILE"

# 对第一个矩阵（如果存在）运行不同方法进行对比
if [ -n "$MATRIX_FILES" ]; then
    FIRST_MATRIX=$(echo "$MATRIX_FILES" | head -n 1)
    if [ -f "$FIRST_MATRIX" ]; then
        echo ""
        echo -e "${YELLOW}对比测试: $(basename "$FIRST_MATRIX")${NC}"
        echo "========================================"

        for method in power jacobi qr; do
            echo ""
            echo -e "${BLUE}方法: $method${NC}"
            echo "----------------------------------------"

            {
                echo ""
                echo "方法: $method"
                echo "----------------------------------------"
            } | tee -a "$RESULT_FILE"

            if [ "$method" == "power" ]; then
                ./eigensolver -f "$FIRST_MATRIX" -m $method -t $TOLERANCE -i $MAX_ITERATIONS 2>&1 | tee -a "$RESULT_FILE"
            else
                ./eigensolver -f "$FIRST_MATRIX" -m $method -t $TOLERANCE -i $MAX_ITERATIONS 2>&1 | tee -a "$RESULT_FILE"
            fi

            echo "" | tee -a "$RESULT_FILE"
        done
    fi
fi

# 最终摘要
{
    echo ""
    echo "========================================"
    echo "性能测试完成！"
    echo "========================================"
    echo "详细结果: $RESULT_FILE"
    echo "结果目录: $RESULT_DIR"
    echo ""
} | tee "$SUMMARY_FILE"

echo -e "${GREEN}性能测试完成！${NC}"
echo ""
echo "详细结果保存在: $RESULT_FILE"
echo ""
