#!/bin/bash
# CrossMedia-PID 环境设置脚本

echo "=========================================="
echo "CrossMedia-PID Anaconda环境设置"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装Anaconda或Miniconda"
    echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda已安装"

# 创建环境
echo ""
echo "📦 创建conda环境 'crossmedia' (Python 3.11)..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "❌ 环境创建失败"
    exit 1
fi

echo "✓ 环境创建成功"

# 激活环境说明
echo ""
echo "=========================================="
echo "🎉 环境设置完成！"
echo "=========================================="
echo ""
echo "激活环境:"
echo "   conda activate crossmedia"
echo ""
echo "运行测试:"
echo "   python experiments/test_api.py"
echo ""
echo "运行主程序:"
echo "   python -m crossmedia_pid.cli process <image_path>"
echo ""
echo "=========================================="
