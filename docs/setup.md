# CrossMedia-PID 环境设置指南

## 方法：使用Anaconda/Miniconda

### 步骤1: 创建环境

在终端中执行：

```bash
cd /Users/sai/Desktop/HKU_Dissertation

# 创建环境（约需5-10分钟）
conda env create -f environment.yml
```

### 步骤2: 激活环境

```bash
conda activate crossmedia
```

### 步骤3: 验证安装

```bash
# 检查Python版本
python --version  # 应显示 Python 3.11.x

# 验证核心依赖
python -c "import httpx, chromadb, numpy, cv2; print('✓ 核心依赖安装成功')"
```

### 步骤4: 配置API密钥

复制示例配置并通过环境变量提供密钥：

```bash
cp configs/config.example.yaml configs/config.yaml
export DASHSCOPE_API_KEY="你的API Key"
```

`configs/config.yaml` 中的核心配置：

```yaml
models:
  vlm:
    provider: "aliyun"
    api_key: "${DASHSCOPE_API_KEY}"
    model_name: "qwen-vl-plus"
```

### 步骤5: 测试API连接

```bash
python experiments/test_api.py
```

### 步骤6: 运行主程序

```bash
# 处理单张图片
python -m crossmedia_pid.cli process <图片路径>

# 批量处理
python -m crossmedia_pid.cli batch <图片目录>

# 查看统计
python -m crossmedia_pid.cli stats
```

---

## 常见问题

### 1. MLX在M1上安装失败

MLX需要M1 Mac，如果安装失败，切换到云服务模式：
```yaml
provider: "aliyun"  # 使用阿里云API
```

### 2. YOLO模型自动下载

模型权重统一放在 `models/`。当前项目已使用 `models/yolov8n.pt` 作为默认路径。

### 3. 向量数据库位置

ChromaDB数据默认存储在项目目录的 `data/chroma_db/` 中。

---

## 环境文件说明

- `environment.yml` - Conda环境定义
- `configs/config.yaml` - 系统配置
- `configs/config.example.yaml` - 不含密钥的配置模板
- `experiments/test_api.py` - API连接测试脚本
