# CrossMedia-PID

跨媒体人物识别原型：从图片或视频帧中检测人物，调用 VLM 抽取人物外观属性，再用稠密向量、稀疏属性向量和 ChromaDB 做身份匹配。

## 目录结构

```text
apps/                 Streamlit 等可视化应用入口
configs/              本地配置和配置模板
crossmedia_pid/       可导入的 Python 包源码
  api/                FastAPI 路由草稿
  core/               检测、VLM 特征、向量化、匹配算法
  db/                 ChromaDB 存储封装
  utils/              属性注册表等通用工具
data/                 本地运行数据，包含 ChromaDB 和属性注册表
docs/                 项目说明、论文材料和截图
experiments/          对比、稳定性、视频匹配等研究脚本
models/               YOLO 等模型权重
scripts/              环境和应用启动脚本
tests/                轻量自动化测试
```

## 快速开始

```bash
conda env create -f environment.yml
conda activate crossmedia
cp configs/config.example.yaml configs/config.yaml
export DASHSCOPE_API_KEY="你的API Key"
```

常用命令：

```bash
python -m crossmedia_pid.cli process <图片路径>
python -m crossmedia_pid.cli batch <图片目录>
python -m crossmedia_pid.cli stats
python experiments/test_api.py
streamlit run apps/video_test_webgui.py
```

也可以使用安装后的命令：

```bash
pip install -e .
crossmedia-pid stats
```

## 架构说明

- `crossmedia_pid/config.py` 负责配置读取、环境变量替换和路径解析。
- `crossmedia_pid/app.py` 负责把检测、特征抽取、向量化和匹配串成完整流水线。
- `crossmedia_pid/cli.py` 只负责命令行交互。
- `apps/video_test_webgui.py` 复用 `CrossMediaPID` 服务，不再复制初始化逻辑。
- GUI 可以从视频轨迹导出测试集，默认写入 `experiments/generated_datasets/`，包含 `images/`、`manifest.csv`、`manifest.jsonl`、`pairs.csv` 和 `summary.json`。
- `experiments/` 下的脚本用于研究验证和性能观察，不作为正式包 API。

本地密钥只放在环境变量或被 git 忽略的 `configs/config.yaml` 中；可提交的模板是 `configs/config.example.yaml`。
