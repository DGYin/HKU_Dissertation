"""
Module C: Dynamic Vectorization
稠密语义向量 + 动态稀疏维度映射
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from crossmedia_pid.config import project_path
from crossmedia_pid.utils.registry import AttributeRegistry, create_sparse_vector, get_registry

logger = logging.getLogger(__name__)


@dataclass
class VectorOutput:
    """C模块输出"""
    dense_vector: List[float]      # 稠密向量
    sparse_vector: Dict[str, float]  # 稀疏向量 {attr_id: weight}
    schema_version: int            # 模式版本
    raw_text: str                  # 用于生成稠密向量的原始文本


class DenseVectorizer:
    """稠密向量生成器（基于BGE模型）"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        onnx_path: Optional[str] = None,
        max_length: int = 512
    ):
        """
        初始化稠密向量生成器
        
        Args:
            model_name: 模型名称或路径
            onnx_path: ONNX模型路径（优先使用）
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.onnx_path = onnx_path
        
        # 延迟加载
        self._session = None
        self._tokenizer = None
        
    def _load_model(self):
        """加载ONNX模型"""
        if self._session is not None:
            return
        
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 加载ONNX模型
            if self.onnx_path and Path(self.onnx_path).exists():
                model_path = self.onnx_path
            else:
                # 尝试从HuggingFace下载ONNX模型
                # 这里简化处理，实际可能需要转换
                from transformers import AutoModel
                logger.warning("ONNX model not found, using transformers (slower)")
                self._model = AutoModel.from_pretrained(self.model_name)
                self._use_transformers = True
                return
            
            # M1优化：使用CoreML或CPU
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._use_transformers = False
            
            logger.info("Embedding model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # 回退到transformers
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._use_transformers = True
            logger.info("Using transformers as fallback")
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2归一化"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def encode(self, text: str) -> np.ndarray:
        """
        编码文本为稠密向量
        
        Args:
            text: 输入文本
            
        Returns:
            稠密向量 (numpy数组)
        """
        self._load_model()
        
        if self._use_transformers:
            # 使用transformers
            import torch
            
            inputs = self._tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # 使用[CLS] token的embedding
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = self._normalize(embeddings.numpy())
            
            return embeddings[0]
        
        else:
            # 使用ONNX
            inputs = self._tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='np'
            )
            
            onnx_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
            
            outputs = self._session.run(None, onnx_inputs)
            embeddings = outputs[0]
            
            # 归一化
            embeddings = self._normalize(embeddings)
            
            return embeddings
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码
        
        Args:
            texts: 文本列表
            
        Returns:
            稠密向量数组 (N, D)
        """
        vectors = []
        for text in texts:
            vec = self.encode(text)
            vectors.append(vec)
        return np.array(vectors)


class DynamicVectorizer:
    """
    动态向量化器
    
    结合稠密向量和稀疏向量
    """
    
    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-zh-v1.5",
        dense_onnx_path: Optional[str] = None,
        max_length: int = 512,
        registry_path: Optional[str] = None
    ):
        """
        初始化动态向量化器
        
        Args:
            dense_model_name: 稠密向量模型
            dense_onnx_path: ONNX模型路径
            max_length: 最大序列长度
            registry_path: 注册表路径
        """
        if registry_path is None:
            registry_path = str(project_path("data", "attribute_registry.json"))

        self.dense_vectorizer = DenseVectorizer(
            model_name=dense_model_name,
            onnx_path=dense_onnx_path,
            max_length=max_length
        )
        self.registry = get_registry(registry_path)
        self.schema_version = 1
    
    def _attributes_to_text(self, attributes: dict) -> str:
        """
        将属性字典转换为文本
        
        用于生成稠密向量
        """
        parts = []
        for key, value in attributes.items():
            if value is None or value == "无":
                continue
            
            # 将键值对转换为描述性文本
            if isinstance(value, bool):
                if value:
                    parts.append(f"{key}")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}:{value}")
            else:
                parts.append(f"{key}是{value}")
        
        return "，".join(parts)
    
    def vectorize(
        self,
        attributes: dict,
        source_meta: Optional[dict] = None
    ) -> VectorOutput:
        """
        向量化属性字典
        
        Args:
            attributes: 属性字典
            source_meta: 源数据元信息
            
        Returns:
            VectorOutput
        """
        # 1. 生成稠密向量
        text = self._attributes_to_text(attributes)
        dense_vec = self.dense_vectorizer.encode(text)
        
        # 2. 生成稀疏向量
        sparse_vec = create_sparse_vector(
            attributes,
            registry=self.registry,
            weight=1.0
        )
        
        return VectorOutput(
            dense_vector=dense_vec.tolist(),
            sparse_vector=sparse_vec,
            schema_version=self.schema_version,
            raw_text=text
        )
    
    def get_registry_stats(self) -> dict:
        """获取注册表统计"""
        return self.registry.get_statistics()


# 便捷函数
def create_vectorizer(config: Optional[dict] = None) -> DynamicVectorizer:
    """从配置创建向量化器"""
    if config is None:
        return DynamicVectorizer()
    
    return DynamicVectorizer(
        dense_model_name=config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
        dense_onnx_path=config.get('onnx_path'),
        max_length=config.get('max_length', 512),
        registry_path=config.get('registry_path', str(project_path("data", "attribute_registry.json")))
    )
