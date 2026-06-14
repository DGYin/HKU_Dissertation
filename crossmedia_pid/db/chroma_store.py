"""
ChromaDB Vector Store
向量数据库封装
"""

import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from crossmedia_pid.config import project_path

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB向量存储封装"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "person_embeddings",
        distance_fn: str = "cosine"
    ):
        """
        初始化ChromaDB存储
        
        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            distance_fn: 距离函数 ('cosine', 'l2', 'ip')
        """
        if persist_directory is None:
            persist_directory = str(project_path("data", "chroma_db"))

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_fn = distance_fn
        
        # 延迟初始化
        self._client = None
        self._collection = None
    
    def _init_client(self):
        """初始化ChromaDB客户端"""
        if self._client is not None:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn}
            )
            
            logger.info(f"ChromaDB collection ready: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add(
        self,
        person_uuid: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        attributes: Dict,
        source_meta: Optional[Dict] = None,
        face_embedding: Optional[List[float]] = None
    ) -> str:
        """
        添加人物向量到数据库
        
        Args:
            person_uuid: 人物UUID
            dense_vector: 稠密向量
            sparse_vector: 稀疏向量（存储在metadata中）
            attributes: 原始属性
            source_meta: 源数据元信息
            face_embedding: 人脸特征（Phase 1暂不使用）
            
        Returns:
            文档ID
        """
        self._init_client()
        
        # 生成唯一文档ID
        doc_id = str(uuid.uuid4())
        
        # 准备metadata
        metadata = {
            "person_uuid": person_uuid,
            "sparse_vector": json.dumps(sparse_vector),
            "attributes": json.dumps(attributes, ensure_ascii=False),
        }
        
        if source_meta:
            metadata["source"] = json.dumps(source_meta, ensure_ascii=False)
        
        if face_embedding:
            metadata["face_embedding"] = json.dumps(face_embedding)
        
        # 添加到集合
        self._collection.add(
            ids=[doc_id],
            embeddings=[dense_vector],
            metadatas=[metadata]
        )
        
        logger.debug(f"Added document {doc_id} for person {person_uuid}")
        
        return doc_id
    
    def search(
        self,
        dense_vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        基于稠密向量搜索
        
        Args:
            dense_vector: 查询稠密向量
            top_k: 返回结果数量
            threshold: 距离阈值（可选）
            
        Returns:
            搜索结果列表
        """
        self._init_client()
        
        results = self._collection.query(
            query_embeddings=[dense_vector],
            n_results=top_k
        )
        
        # 格式化结果
        candidates = []
        
        if not results['ids'] or not results['ids'][0]:
            return candidates
        
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i] if results['distances'] else None
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            # ChromaDB cosine距离是1-cosine_similarity
            similarity = 1 - distance if distance is not None else 0
            
            # 应用阈值
            if threshold is not None and similarity < threshold:
                continue
            
            candidate = {
                'doc_id': doc_id,
                'person_uuid': metadata.get('person_uuid'),
                'dense_similarity': similarity,
                'distance': distance,
                'sparse_vector': json.loads(metadata.get('sparse_vector', '{}')),
                'attributes': json.loads(metadata.get('attributes', '{}')),
                'source': json.loads(metadata.get('source', '{}'))
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def get_by_person_uuid(self, person_uuid: str) -> List[Dict]:
        """
        通过PersonUUID获取所有记录
        
        Args:
            person_uuid: 人物UUID
            
        Returns:
            记录列表
        """
        self._init_client()
        
        results = self._collection.get(
            where={"person_uuid": person_uuid}
        )
        
        records = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            record = {
                'doc_id': doc_id,
                'embedding': results['embeddings'][i] if results['embeddings'] else None,
                'person_uuid': metadata.get('person_uuid'),
                'sparse_vector': json.loads(metadata.get('sparse_vector', '{}')),
                'attributes': json.loads(metadata.get('attributes', '{}')),
                'source': json.loads(metadata.get('source', '{}'))
            }
            records.append(record)
        
        return records
    
    def count(self) -> int:
        """获取记录总数"""
        self._init_client()
        return self._collection.count()
    
    def get_all_person_uuids(self) -> List[str]:
        """获取所有唯一的PersonUUID"""
        self._init_client()
        
        results = self._collection.get()
        
        uuids = set()
        for metadata in results['metadatas']:
            uuids.add(metadata.get('person_uuid'))
        
        return list(uuids)
    
    def delete_person(self, person_uuid: str):
        """
        删除指定人物的所有记录
        
        Args:
            person_uuid: 人物UUID
        """
        self._init_client()
        
        self._collection.delete(
            where={"person_uuid": person_uuid}
        )
        
        logger.info(f"Deleted all records for person {person_uuid}")


def create_chroma_store(config: Optional[dict] = None) -> ChromaStore:
    """从配置创建ChromaStore"""
    if config is None:
        return ChromaStore()
    
    return ChromaStore(
        persist_directory=config.get('persist_directory', str(project_path("data", "chroma_db"))),
        collection_name=config.get('collection_name', 'person_embeddings'),
        distance_fn=config.get('distance_fn', 'cosine')
    )
