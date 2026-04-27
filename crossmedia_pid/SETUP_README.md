# CrossMedia-PID 环境设置指南

## 方法：使用Anaconda/Miniconda

### 步骤1: 创建环境

在终端中执行：

```bash
cd /Users/sai/Desktop/HKU_Dissertation/crossmedia_pid

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

编辑 `configs/config.yaml`：

```yaml
models:
  vlm:
    provider: "aliyun"
    api_key: "sk-03dc0f75f97e45e9a4be3a28c0107ac7"  # 你的API Key
    model_name: "Qwen3-VL-Flash"
```

### 步骤5: 测试API连接

```bash
python test_api.py
```

### 步骤6: 运行主程序

```bash
# 处理单张图片
python main.py process <图片路径>

# 批量处理
python main.py batch <图片目录>

# 查看统计
python main.py stats
```

---

## 常见问题

### 1. MLX在M1上安装失败

MLX需要M1 Mac，如果安装失败，切换到云服务模式：
```yaml
provider: "aliyun"  # 使用阿里云API
```

### 2. YOLO模型自动下载

首次运行时会自动下载 `yolov8n.pt` 到当前目录。

### 3. 向量数据库位置

ChromaDB数据默认存储在项目目录的 `./chroma_db/` 中。

---

## 环境文件说明

- `environment.yml` - Conda环境定义
- `configs/config.yaml` - 系统配置
- `test_api.py` - API连接测试脚本
