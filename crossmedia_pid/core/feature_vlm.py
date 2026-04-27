"""
Module B: Open-domain Feature Extraction
使用VLM动态提取人物特征
支持本地MLX和云服务API两种模式
"""

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# VLM Prompt模板
PERSON_FEATURE_PROMPT = """你是一个专业的人物特征分析专家，专门用于监控视频分析。请仔细观察输入图片中的人物，提取详细的结构化特征并严格返回JSON。

【必须提取的特征类目】：
- gender: 性别（男/女/未知）
- age_group: 年龄段（儿童/青年/中年/老年）
- height_build: 身高体型（高瘦/中等/矮胖等）

【上衣详细特征 - 重点关注】：
- topwear_color: 上衣主颜色
- topwear_type: 上衣类型（T恤/衬衫/外套/卫衣/夹克等）
- topwear_pattern_type: 图案类型（纯色/条纹/格子/印花/图形/文字/logo等）
- topwear_pattern_desc: 图案详细描述（如：胸前有白色大logo、袖子有条纹、背后有图案等）
- topwear_pattern_position: 图案位置（胸前/背后/左胸/右胸/袖子/下摆等，图案的起始点与终止点）
- topwear_pattern_size: 图案大小（大/中/小/占据整个衣服等）
- topwear_pattern_color: 图案颜色

【下装详细特征】：
- bottomwear_color: 下装颜色
- bottomwear_type: 下装类型（长裤/短裤/裙子等）
- bottomwear_pattern: 下装图案描述（纯色/条纹/有装饰等）

【鞋子特征】：
- shoes_color: 鞋子颜色
- shoes_type: 鞋子类型（运动鞋/皮鞋/凉鞋/靴子等）

【配饰与携带物】：
- bag: 是否携带包（无/背包/手提包/挎包等，描述颜色和类型）
- glasses: 是否戴眼镜（无/眼镜类型和颜色）
- hat: 是否戴帽子（无/帽子类型和颜色）
- mask: 是否戴口罩

【头部特征】：
- hair_style: 发型（短发/长发/卷发/直发/光头等）
- hair_color: 发色（黑色/棕色/金色/灰色等）

【显著特征】：
- distinctive_traits: 显著特征（纹身位置、疤痕、特殊标记等）
- holding_object: 手持物品
- posture: 姿态（站立/行走/坐着/蹲着等）

【规则】
1. 对于衣服图案，如有，必须详细描述：图案形状、位置、大小、颜色
2. 如果看到logo、文字、标志，必须描述其内容和位置
3. 若某项特征无法确定，值请严格填为 "无"
4. 仅输出合法JSON，不要包含任何解释性文字
5. 值使用中文描述，键名使用英文snake_case

请分析图片并输出JSON："""


@dataclass
class FeatureOutput:
    """B模块输出"""
    attributes: Dict[str, Any]  # 清洗后的动态特征字典
    raw_response: str           # VLM原始输出（用于调试）
    is_valid: bool              # 是否成功解析
    provider: str               # 使用的提供商 (mlx/cloud)


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""
    
    def __init__(self, prompt: Optional[str] = None):
        self.prompt = prompt or PERSON_FEATURE_PROMPT
    
    @abstractmethod
    def extract(self, image: np.ndarray, apply_filter: bool = True) -> FeatureOutput:
        """提取特征"""
        pass
    
    def _normalize_key(self, key: str) -> str:
        """标准化键名"""
        key = key.lower().strip()
        key = re.sub(r'\s+', '_', key)
        key = re.sub(r'[^a-z0-9_]', '', key)
        return key
    
    def _clean_value(self, value: Any) -> Any:
        """清洗值"""
        if value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if value in ["无", "", "null", "NULL", "None", "none"]:
                return None
        
        return value
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析VLM的JSON响应"""
        response = response.strip()
        
        # 尝试1: 直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试2: 提取代码块
        code_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # 尝试3: 使用json_repair
        try:
            import json_repair
            repaired = json_repair.repair_json(response, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            logger.warning(f"json_repair failed: {e}")
        
        # 尝试4: 手动提取
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Failed to parse JSON from response: {response[:200]}...")
    
    def _filter_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """过滤属性"""
        cleaned = {}
        
        for key, value in attributes.items():
            norm_key = self._normalize_key(key)
            if not norm_key:
                continue
            
            cleaned_value = self._clean_value(value)
            if cleaned_value is None:
                continue
            
            cleaned[norm_key] = cleaned_value
        
        return cleaned
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将numpy数组转换为base64编码"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    
    def _normalize_key(self, key: str) -> str:
        """
        标准化键名
        - 转小写
        - 替换空格为下划线
        - 移除特殊字符
        """
        key = key.lower().strip()
        key = re.sub(r'\s+', '_', key)
        key = re.sub(r'[^a-z0-9_]', '', key)
        return key
    
    def _clean_value(self, value: Any) -> Any:
        """
        清洗值
        - 字符串"无"、null、空字符串 → 标记为过滤
        - 布尔值保留
        - 数值保留
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if value in ["无", "", "null", "NULL", "None", "none"]:
                return None
        
        return value
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        解析VLM的JSON响应
        
        策略：
        1. 尝试直接解析整个响应
        2. 尝试提取代码块中的JSON
        3. 尝试修复常见JSON错误
        """
        # 清理响应
        response = response.strip()
        
        # 尝试1: 直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试2: 提取代码块
        code_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}'  # 最宽松的匹配
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # 尝试3: 使用json_repair
        try:
            import json_repair
            repaired = json_repair.repair_json(response, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            logger.warning(f"json_repair failed: {e}")
        
        # 尝试4: 手动提取键值对
        try:
            # 尝试找到最外层的大括号
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end+1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        raise ValueError(f"Failed to parse JSON from response: {response[:200]}...")
    
    def _filter_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤属性
        - 标准化键名
        - 过滤"无"值
        - 保留有效特征
        """
        cleaned = {}
        
        for key, value in attributes.items():
            # 标准化键名
            norm_key = self._normalize_key(key)
            if not norm_key:
                continue
            
            # 清洗值
            cleaned_value = self._clean_value(value)
            if cleaned_value is None:
                continue
            
            cleaned[norm_key] = cleaned_value
        
        return cleaned
    
class MLXFeatureExtractor(BaseFeatureExtractor):
    """本地MLX特征提取器"""
    
    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-VL-235B-4bit",
        max_tokens: int = 512,
        temperature: float = 0.1,
        prompt: Optional[str] = None
    ):
        super().__init__(prompt)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._model = None
        self._processor = None
    
    def _load_model(self):
        """延迟加载MLX VLM模型"""
        if self._model is not None:
            return
        
        try:
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template
            
            logger.info(f"Loading MLX VLM model: {self.model_name}")
            
            self._model, self._processor = load(self.model_name)
            self._apply_chat_template = apply_chat_template
            
            logger.info("MLX VLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MLX VLM model: {e}")
            raise
    
    def extract(self, image: np.ndarray, apply_filter: bool = True) -> FeatureOutput:
        """使用MLX本地模型提取特征"""
        self._load_model()
        
        # 转换图片格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # 准备消息
        messages = [{"role": "user", "content": self.prompt}]
        prompt_text = self._apply_chat_template(
            self._processor,
            self._model.config,
            messages
        )
        
        try:
            from mlx_vlm import generate
            
            response = generate(
                self._model,
                self._processor,
                prompt_text,
                image=pil_image,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                verbose=False
            )
            
            try:
                attributes = self._parse_json_response(response)
                if apply_filter:
                    attributes = self._filter_attributes(attributes)
                
                return FeatureOutput(
                    attributes=attributes,
                    raw_response=response,
                    is_valid=True,
                    provider="mlx"
                )
                
            except ValueError as e:
                logger.error(f"JSON parsing failed: {e}")
                return FeatureOutput(
                    attributes={},
                    raw_response=response,
                    is_valid=False,
                    provider="mlx"
                )
        
        except Exception as e:
            logger.error(f"MLX inference failed: {e}")
            return FeatureOutput(
                attributes={},
                raw_response=str(e),
                is_valid=False,
                provider="mlx"
            )


class CloudFeatureExtractor(BaseFeatureExtractor):
    """云服务API特征提取器
    
    支持OpenAI兼容格式的API（如OpenAI、DeepSeek、SiliconFlow等）
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "gpt-4o",
        max_tokens: int = 512,
        temperature: float = 0.1,
        prompt: Optional[str] = None,
        timeout: int = 60
    ):
        super().__init__(prompt)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        self._client = None
    
    def _get_client(self):
        """获取或创建HTTP客户端"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("httpx is required for cloud API. Install with: pip install httpx")
        return self._client
    
    def _build_payload(self, base64_image: str) -> dict:
        """构建API请求体（可被阿里云类覆盖）"""
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def _parse_response(self, result: dict) -> str:
        """解析API响应（可被阿里云类覆盖）"""
        return result["choices"][0]["message"]["content"]
    
    def extract(self, image: np.ndarray, apply_filter: bool = True) -> FeatureOutput:
        """使用云服务API提取特征"""
        try:
            client = self._get_client()
            
            # 将图片转换为base64
            base64_image = self._image_to_base64(image)
            
            # 构建请求
            payload = self._build_payload(base64_image)
            
            logger.debug(f"Sending request to {self.base_url}/chat/completions")
            
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = self._parse_response(result)
            
            try:
                attributes = self._parse_json_response(content)
                if apply_filter:
                    attributes = self._filter_attributes(attributes)
                
                return FeatureOutput(
                    attributes=attributes,
                    raw_response=content,
                    is_valid=True,
                    provider="cloud"
                )
                
            except ValueError as e:
                logger.error(f"JSON parsing failed: {e}")
                return FeatureOutput(
                    attributes={},
                    raw_response=content,
                    is_valid=False,
                    provider="cloud"
                )
        
        except Exception as e:
            logger.error(f"Cloud API request failed: {e}")
            return FeatureOutput(
                attributes={},
                raw_response=str(e),
                is_valid=False,
                provider="cloud"
            )


class AliyunFeatureExtractor(BaseFeatureExtractor):
    """阿里云DashScope特征提取器
    
    阿里云DashScope API使用OpenAI兼容格式:
    - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
    - API Key: 通过Authorization Header传递
    
    文档: https://help.aliyun.com/document_detail/2781831.html
    """
    
    # 阿里云DashScope视觉语言模型列表
    # 文档: https://help.aliyun.com/document_detail/2781831.html
    #  pricing: https://dashscope.aliyun.com/pricing
    SUPPORTED_MODELS = {
        # ===== 通义千问VL系列 (推荐) =====
        "qwen-vl-max": "通义千问VL Max - 最强视觉理解能力",
        "qwen-vl-max-latest": "通义千问VL Max最新版 - 始终使用最新版本",
        "qwen-vl-plus": "通义千问VL Plus - 性价比之选",
        
        # ===== Qwen2.5-VL系列 (开源新版) =====
        "qwen2.5-vl-72b-instruct": "Qwen2.5 VL 72B - 开源大参数版本",
        "qwen2.5-vl-7b-instruct": "Qwen2.5 VL 7B - 开源轻量版本",
        "qwen2.5-vl-3b-instruct": "Qwen2.5 VL 3B - 开源超轻量版本",
        
        # ===== Qwen2-VL系列 (旧版但仍可用) =====
        "qwen-vl-chat-v1": "通义千问VL Chat V1 - 早期版本",
        
        # ===== 实验/预览版模型 =====
        # "qwen3-vl-xxx": "Qwen3 VL系列 - 待发布",
    }
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-vl-max",
        max_tokens: int = 512,
        temperature: float = 0.1,
        prompt: Optional[str] = None,
        timeout: int = 60,
        enable_search: bool = False,
        enable_thinking: bool = False
    ):
        super().__init__(prompt)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.enable_search = enable_search
        self.enable_thinking = enable_thinking
        
        # 阿里云OpenAI兼容模式endpoint
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        self._client = None
        
        # 验证模型
        # if model_name not in self.SUPPORTED_MODELS:
        #     logger.warning(
        #         f"Model '{model_name}' not in known models. "
        #         f"Known models: {list(self.SUPPORTED_MODELS.keys())}"
        #     )
    
    def _get_client(self):
        """获取HTTP客户端"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("httpx is required. Install with: pip install httpx")
        return self._client
    
    def extract(self, image: np.ndarray, apply_filter: bool = True) -> FeatureOutput:
        """使用阿里云DashScope API提取特征"""
        try:
            client = self._get_client()
            
            # 转换图片为base64
            base64_image = self._image_to_base64(image)
            
            # 构建请求体 (OpenAI兼容格式)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            # 添加阿里云特有参数
            if self.enable_search:
                payload["enable_search"] = True
            
            # enable_thinking 需要放在 parameters 对象中
            if self.enable_thinking:
                if "parameters" not in payload:
                    payload["parameters"] = {}
                payload["parameters"]["enable_thinking"] = True
            
            logger.debug(f"Sending request to {self.base_url}/chat/completions")
            logger.debug(f"Model: {self.model_name}")
            
            # 发送请求
            response = client.post("/chat/completions", json=payload)
            
            # 检查响应
            if response.status_code == 404:
                logger.error(f"Model '{self.model_name}' not found or not available")
                logger.error(f"Response: {response.text}")
                return FeatureOutput(
                    attributes={},
                    raw_response=f"Model not found: {response.text}",
                    is_valid=False,
                    provider="aliyun"
                )
            
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析JSON
            try:
                attributes = self._parse_json_response(content)
                if apply_filter:
                    attributes = self._filter_attributes(attributes)
                
                return FeatureOutput(
                    attributes=attributes,
                    raw_response=content,
                    is_valid=True,
                    provider="aliyun"
                )
                
            except ValueError as e:
                logger.error(f"JSON parsing failed: {e}")
                return FeatureOutput(
                    attributes={},
                    raw_response=content,
                    is_valid=False,
                    provider="aliyun"
                )
        
        except Exception as e:
            logger.error(f"Aliyun API request failed: {e}")
            return FeatureOutput(
                attributes={},
                raw_response=str(e),
                is_valid=False,
                provider="aliyun"
            )


class FeatureExtractor:
    """统一特征提取器接口
    
    根据配置自动选择本地MLX或云服务API
    """
    
    def __init__(
        self,
        provider: str = "cloud",  # "mlx", "cloud", "aliyun"
        # MLX参数
        model_name: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        prompt: Optional[str] = None,
        # Cloud参数
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        # 阿里云特有参数
        enable_search: bool = False,
        enable_thinking: bool = False
    ):
        """
        初始化特征提取器
        
        Args:
            provider: 提供商类型 ("mlx", "cloud", "aliyun")
            model_name: 模型名称
            max_tokens: 最大生成token数
            temperature: 采样温度
            prompt: 自定义prompt
            api_key: 云服务API密钥
            base_url: 云服务API基础URL (cloud模式需要)
            timeout: API请求超时时间
            enable_search: 阿里云联网搜索功能
            enable_thinking: 阿里云思考模式功能
        """
        self.provider = provider
        
        if provider == "mlx":
            self._extractor = MLXFeatureExtractor(
                model_name=model_name or "mlx-community/Qwen3-VL-235B-4bit",
                max_tokens=max_tokens,
                temperature=temperature,
                prompt=prompt
            )
        elif provider == "cloud":
            if not api_key or not base_url:
                raise ValueError("Cloud provider requires api_key and base_url")
            
            self._extractor = CloudFeatureExtractor(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name or "gpt-4o",
                max_tokens=max_tokens,
                temperature=temperature,
                prompt=prompt,
                timeout=timeout
            )
        elif provider == "aliyun":
            if not api_key:
                raise ValueError("Aliyun provider requires api_key")
            
            self._extractor = AliyunFeatureExtractor(
                api_key=api_key,
                model_name=model_name or "qwen-vl-max",
                max_tokens=max_tokens,
                temperature=temperature,
                prompt=prompt,
                timeout=timeout,
                enable_search=enable_search,
                enable_thinking=enable_thinking
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'mlx', 'cloud', or 'aliyun'")
    
    def extract(self, image: np.ndarray, apply_filter: bool = True) -> FeatureOutput:
        """提取特征"""
        return self._extractor.extract(image, apply_filter)
    
    def extract_batch(self, images: list, apply_filter: bool = True) -> list:
        """批量提取特征"""
        results = []
        for img in images:
            result = self.extract(img, apply_filter)
            results.append(result)
        return results


def create_feature_extractor(config: Optional[dict] = None) -> FeatureExtractor:
    """从配置创建特征提取器
    
    配置示例:
    
    # 本地MLX模式
    {
        "provider": "mlx",
        "model_name": "mlx-community/Qwen3-VL-235B-4bit",
        "max_tokens": 512,
        "temperature": 0.1
    }
    
    # 云服务模式 (OpenAI兼容)
    {
        "provider": "cloud",
        "api_key": "sk-xxx",
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o",
        "max_tokens": 512,
        "temperature": 0.1
    }
    
    # 阿里云模式
    {
        "provider": "aliyun",
        "api_key": "sk-xxx",
        "model_name": "qwen-vl-max",
        "max_tokens": 512,
        "temperature": 0.1,
        "enable_search": false,
        "enable_thinking": false
    }
    """
    if config is None:
        return FeatureExtractor()
    
    return FeatureExtractor(
        provider=config.get('provider', 'cloud'),
        model_name=config.get('model_name'),
        max_tokens=config.get('max_tokens', 512),
        temperature=config.get('temperature', 0.1),
        prompt=config.get('prompt'),
        api_key=config.get('api_key'),
        base_url=config.get('base_url'),
        timeout=config.get('timeout', 60),
        enable_search=config.get('enable_search', False),
        enable_thinking=config.get('enable_thinking', False)
    )
