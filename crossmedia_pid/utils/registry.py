"""
Attribute Registry: 动态属性注册表
管理稀疏向量的维度映射
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AttributeRegistry:
    """
    属性注册表
    
    维护属性字符串到ID的映射，支持动态扩展
    持久化到JSON文件
    """
    
    def __init__(self, persist_path: str = "./attribute_registry.json"):
        """
        初始化注册表
        
        Args:
            persist_path: 持久化文件路径
        """
        self.persist_path = Path(persist_path)
        self._lock = threading.Lock()
        
        # 内存中的注册表
        self._registry: Dict[str, dict] = {}
        self._next_id: int = 1
        
        # 加载已有数据
        self._load()
    
    def _load(self):
        """从文件加载注册表"""
        if not self.persist_path.exists():
            logger.info(f"Registry file not found, creating new: {self.persist_path}")
            self._save()
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._registry = data.get('registry', {})
            self._next_id = data.get('next_id', 1)
            
            # 确保next_id正确
            if self._registry:
                max_id = max(info['id'] for info in self._registry.values())
                self._next_id = max(self._next_id, max_id + 1)
            
            logger.info(f"Loaded registry: {len(self._registry)} attributes, next_id={self._next_id}")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}, creating new")
            self._registry = {}
            self._next_id = 1
    
    def _save(self):
        """保存注册表到文件"""
        data = {
            'registry': self._registry,
            'next_id': self._next_id,
            'updated_at': time.time()
        }
        
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register(self, attr_key: str, auto_save: bool = True) -> int:
        """
        注册属性，返回属性ID
        
        Args:
            attr_key: 属性键名
            auto_save: 是否自动保存
            
        Returns:
            属性ID
        """
        with self._lock:
            if attr_key in self._registry:
                # 已存在，增加计数
                self._registry[attr_key]['count'] += 1
                self._registry[attr_key]['last_seen'] = time.time()
                attr_id = self._registry[attr_key]['id']
            else:
                # 新属性，分配ID
                attr_id = self._next_id
                self._registry[attr_key] = {
                    'id': attr_id,
                    'key': attr_key,
                    'count': 1,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
                self._next_id += 1
                logger.info(f"New attribute registered: '{attr_key}' -> ID {attr_id}")
            
            if auto_save:
                self._save()
            
            return attr_id
    
    def get_id(self, attr_key: str) -> Optional[int]:
        """
        获取属性ID
        
        Args:
            attr_key: 属性键名
            
        Returns:
            属性ID，如果不存在返回None
        """
        with self._lock:
            info = self._registry.get(attr_key)
            return info['id'] if info else None
    
    def get_key(self, attr_id: int) -> Optional[str]:
        """
        通过ID获取属性键名
        
        Args:
            attr_id: 属性ID
            
        Returns:
            属性键名，如果不存在返回None
        """
        with self._lock:
            for key, info in self._registry.items():
                if info['id'] == attr_id:
                    return key
            return None
    
    def get_info(self, attr_key: str) -> Optional[dict]:
        """
        获取属性完整信息
        
        Args:
            attr_key: 属性键名
            
        Returns:
            属性信息字典
        """
        with self._lock:
            return self._registry.get(attr_key)
    
    def get_all_keys(self) -> list:
        """获取所有属性键名"""
        with self._lock:
            return list(self._registry.keys())
    
    def get_verified_keys(self, min_frequency: int = 3) -> list:
        """
        获取已验证的属性键名（频率>=阈值）
        
        Args:
            min_frequency: 最小频率阈值
            
        Returns:
            已验证属性列表
        """
        with self._lock:
            return [
                key for key, info in self._registry.items()
                if info['count'] >= min_frequency
            ]
    
    def get_statistics(self) -> dict:
        """获取注册表统计信息"""
        with self._lock:
            total = len(self._registry)
            verified = len(self.get_verified_keys())
            return {
                'total_attributes': total,
                'verified_attributes': verified,
                'next_id': self._next_id
            }
    
    def reset(self, confirm: bool = False):
        """
        重置注册表（危险操作）
        
        Args:
            confirm: 必须传入True才会执行
        """
        if not confirm:
            logger.warning("Registry reset skipped (confirm=False)")
            return
        
        with self._lock:
            self._registry = {}
            self._next_id = 1
            self._save()
            logger.warning("Registry has been reset!")


# 全局注册表实例（单例模式）
_registry_instance: Optional[AttributeRegistry] = None
_registry_lock = threading.Lock()


def get_registry(persist_path: str = "./attribute_registry.json") -> AttributeRegistry:
    """
    获取全局注册表实例
    
    Args:
        persist_path: 持久化路径
        
    Returns:
        AttributeRegistry实例
    """
    global _registry_instance
    
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = AttributeRegistry(persist_path)
        return _registry_instance


def create_sparse_vector(
    attributes: dict,
    registry: Optional[AttributeRegistry] = None,
    weight: float = 1.0
) -> dict:
    """
    从属性字典创建稀疏向量
    
    Args:
        attributes: 属性字典
        registry: 注册表实例（默认使用全局）
        weight: 默认权重
        
    Returns:
        稀疏向量 {attr_id: weight}
    """
    if registry is None:
        registry = get_registry()
    
    sparse_vec = {}
    
    for key, value in attributes.items():
        # 跳过无效值
        if value is None or value == "无":
            continue
        
        # 注册并获取ID
        attr_id = registry.register(key)
        
        # 计算权重（可以根据值调整）
        # 简单实现：所有属性权重相同
        attr_weight = weight
        
        sparse_vec[str(attr_id)] = attr_weight
    
    return sparse_vec
