# 阿里云DashScope视觉语言模型参考

## 快速选择指南

| 使用场景 | 推荐模型 | 说明 |
|---------|---------|------|
| **最佳效果** | `qwen-vl-max` | 最强视觉理解能力，适合复杂场景 |
| **性价比** | `qwen-vl-plus` | 平衡效果与成本，推荐日常使用 |
| **自动更新** | `qwen-vl-max-latest` | 始终使用最新版本 |
| **开源方案** | `qwen2.5-vl-72b-instruct` | 开源可部署，大参数版本 |
| **轻量快速** | `qwen2.5-vl-7b-instruct` | 开源轻量版，响应快 |

---

## 完整模型列表

### 1. 通义千问VL系列 (商业版/推荐)

| 模型ID | 名称 | 特点 | 适用场景 |
|--------|------|------|----------|
| `qwen-vl-max` | 通义千问VL Max | 最强视觉理解，多模态推理 | 复杂图像分析、专业场景 |
| `qwen-vl-max-latest` | VL Max最新版 | 自动指向最新版本 | 生产环境，自动更新 |
| `qwen-vl-plus` | 通义千问VL Plus | 性价比高，效果优秀 | 日常开发、批量处理 |

### 2. Qwen2.5-VL系列 (开源新版)

| 模型ID | 参数量 | 特点 | 适用场景 |
|--------|--------|------|----------|
| `qwen2.5-vl-72b-instruct` | 72B | 开源最强，性能接近商业版 | 私有化部署、研究 |
| `qwen2.5-vl-7b-instruct` | 7B | 轻量快速，效果良好 | 边缘设备、实时应用 |
| `qwen2.5-vl-3b-instruct` | 3B | 超轻量，极速响应 | 移动端、低延迟场景 |

### 3. 历史版本 (仍可用但不推荐)

| 模型ID | 说明 |
|--------|------|
| `qwen-vl-chat-v1` | 早期版本，建议迁移到新版 |

---

## 配置示例

### 推荐配置 (性价比)
```yaml
models:
  vlm:
    provider: "aliyun"
    api_key: "${DASHSCOPE_API_KEY}"
    model_name: "qwen-vl-plus"  # 性价比之选
    max_tokens: 512
    temperature: 0.1
```

### 最佳效果配置
```yaml
models:
  vlm:
    provider: "aliyun"
    api_key: "${DASHSCOPE_API_KEY}"
    model_name: "qwen-vl-max"  # 最强效果
    max_tokens: 512
    temperature: 0.1
```

### 开源模型配置 (如需本地部署)
```yaml
models:
  vlm:
    provider: "aliyun"
    api_key: "${DASHSCOPE_API_KEY}"
    model_name: "qwen2.5-vl-7b-instruct"  # 开源轻量版
    max_tokens: 512
    temperature: 0.1
```

---

## 参考链接

- **官方文档**: https://help.aliyun.com/document_detail/2781831.html
- **价格详情**: https://dashscope.aliyun.com/pricing
- **控制台**: https://dashscope.aliyun.com/console

---

## 注意事项

1. **模型可用性**: 部分模型可能需要申请开通
2. **地域限制**: 确保在支持的region使用
3. **配额限制**: 新用户有免费额度，注意查看
4. **版本更新**: `qwen-vl-max-latest` 会自动更新，生产环境建议固定版本

---

*最后更新: 2025年1月*
