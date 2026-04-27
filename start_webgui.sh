#!/bin/bash
# CrossMedia-PID Web GUI 启动脚本

# 设置环境变量禁用Streamlit邮件收集
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

lsof -ti:8501 | xargs kill -9 2>/dev/null; sleep 1; echo "Port 8501 cleared"

echo "🚀 启动 CrossMedia-PID Web GUI..."
echo "📍 工作目录: $SCRIPT_DIR"
echo ""

# 使用conda运行streamlit
conda run -n crossmedia streamlit run video_test_webgui.py \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.port=8501 \
    --server.address=localhost

# 如果上面的命令失败，尝试备用方式
if [ $? -ne 0 ]; then
    echo "尝试备用启动方式..."
    conda run -n crossmedia python -c "
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
import sys
sys.argv = ['streamlit', 'run', 'video_test_webgui.py', '--server.headless=true', '--browser.gatherUsageStats=false']
import streamlit.web.cli
streamlit.web.cli.main()
"
fi
