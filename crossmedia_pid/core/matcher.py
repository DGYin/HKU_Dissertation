"""
Module D: Identity Matching
混合距离计算 + 身份决策
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from db.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class MatchOutput:
    """D模块输出"""
    person_uuid: str
    match_score: float
    is_new_identity: bool
    top_candidates: List[Dict]
    dense_score: float
    sparse_score: float
    face_score: Optional[float]


class IdentityMatcher:
    """身份匹配器"""
    
    def __init__(
        self,
        store: ChromaStore,
        threshold: float = 0.72,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        enable_face: bool = False
    ):
        """
        初始化身份匹配器
        
        Args:
            store: ChromaDB存储
            threshold: 匹配阈值
            top_k: 检索候选数
            weights: 距离权重 {'dense', 'sparse', 'face'}
            enable_face: 是否启用人脸特征（Phase 1=False）
        """
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.enable_face = enable_face
        
        # 默认权重
        if weights is None:
            if enable_face:
                weights = {'dense': 0.25, 'sparse': 0.15, 'face': 0.6}
            else:
                weights = {'dense': 0.65, 'sparse': 0.35, 'face': 0.0}
        
        self.weights = weights
        
        # 验证权重
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            self.weights = {k: v/total for k, v in weights.items()}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def _jaccard_similarity(
        self,
        sparse1: Dict[str, float],
        sparse2: Dict[str, float]
    ) -> float:
        """
        计算Jaccard相似度
        
        对于稀疏向量，计算共有属性 / 总属性
        """
        if not sparse1 or not sparse2:
            return 0.0
        
        keys1 = set(sparse1.keys())
        keys2 = set(sparse2.keys())
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        if not union:
            return 0.0
        
        # 加权Jaccard
        intersection_weight = sum(
            min(sparse1.get(k, 0), sparse2.get(k, 0))
            for k in intersection
        )
        union_weight = sum(
            max(sparse1.get(k, 0), sparse2.get(k, 0))
            for k in union
        )
        
        if union_weight == 0:
            return 0.0
        
        return intersection_weight / union_weight
    
    def _calculate_total_score(
        self,
        dense_score: float,
        sparse_score: float,
        face_score: Optional[float] = None
    ) -> float:
        """
        计算综合匹配分数
        
        S_total = w_d * dense + w_s * sparse + w_f * face
        """
        total = self.weights['dense'] * dense_score
        total += self.weights['sparse'] * sparse_score
        
        if self.enable_face and face_score is not None:
            total += self.weights['face'] * face_score
        
        return total
    
    def match(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        face_embedding: Optional[List[float]] = None,
        query_attributes: Optional[Dict] = None
    ) -> MatchOutput:
        """
        执行身份匹配
        
        Args:
            dense_vector: 查询稠密向量
            sparse_vector: 查询稀疏向量
            face_embedding: 查询人脸特征（可选）
            query_attributes: 查询属性（用于日志）
            
        Returns:
            MatchOutput
        """
        # 1. 检索候选
        candidates = self.store.search(
            dense_vector=dense_vector,
            top_k=self.top_k
        )
        
        if not candidates:
            # 没有候选，创建新身份
            new_uuid = f"p_{uuid.uuid4().hex[:8]}"
            logger.info(f"No candidates found, creating new identity: {new_uuid}")
            
            return MatchOutput(
                person_uuid=new_uuid,
                match_score=0.0,
                is_new_identity=True,
                top_candidates=[],
                dense_score=0.0,
                sparse_score=0.0,
                face_score=None
            )
        
        # 2. 计算混合距离
        scored_candidates = []
        
        for cand in candidates:
            # 稠密向量相似度（已从ChromaDB获取）
            dense_sim = cand['dense_similarity']
            
            # 稀疏向量Jaccard相似度
            sparse_sim = self._jaccard_similarity(
                sparse_vector,
                cand['sparse_vector']
            )
            
            # 人脸相似度（Phase 1跳过）
            face_sim = None
            if self.enable_face and face_embedding and 'face_embedding' in cand:
                face_sim = self._cosine_similarity(
                    face_embedding,
                    cand['face_embedding']
                )
            
            # 综合分数
            total_score = self._calculate_total_score(dense_sim, sparse_sim, face_sim)
            
            scored_candidates.append({
                **cand,
                'sparse_similarity': sparse_sim,
                'face_similarity': face_sim,
                'total_score': total_score
            })
        
        # 3. 按综合分数排序
        scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 4. 决策
        best_candidate = scored_candidates[0]
        best_score = best_candidate['total_score']
        
        if best_score >= self.threshold:
            # 匹配成功，归并到已有身份
            matched_uuid = best_candidate['person_uuid']
            logger.info(
                f"Matched to existing identity: {matched_uuid} "
                f"(score={best_score:.3f}, threshold={self.threshold})"
            )
            
            return MatchOutput(
                person_uuid=matched_uuid,
                match_score=best_score,
                is_new_identity=False,
                top_candidates=scored_candidates[:3],
                dense_score=best_candidate['dense_similarity'],
                sparse_score=best_candidate['sparse_similarity'],
                face_score=best_candidate.get('face_similarity')
            )
        
        else:
            # 未达阈值，创建新身份
            new_uuid = f"p_{uuid.uuid4().hex[:8]}"
            logger.info(
                f"Best score {best_score:.3f} below threshold {self.threshold}, "
                f"creating new identity: {new_uuid}"
            )
            
            return MatchOutput(
                person_uuid=new_uuid,
                match_score=best_score,
                is_new_identity=True,
                top_candidates=scored_candidates[:3],
                dense_score=best_candidate['dense_similarity'],
                sparse_score=best_candidate['sparse_similarity'],
                face_score=best_candidate.get('face_similarity')
            )
    
    def add_identity(
        self,
        person_uuid: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        attributes: Dict,
        source_meta: Optional[Dict] = None,
        face_embedding: Optional[List[float]] = None
    ) -> str:
        """
        添加身份到数据库
        
        Args:
            person_uuid: 人物UUID
            dense_vector: 稠密向量
            sparse_vector: 稀疏向量
            attributes: 属性字典
            source_meta: 源数据元信息
            face_embedding: 人脸特征
            
        Returns:
            文档ID
        """
        doc_id = self.store.add(
            person_uuid=person_uuid,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            attributes=attributes,
            source_meta=source_meta,
            face_embedding=face_embedding
        )
        
        return doc_id
    
    def search_similar(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        搜索相似人物（不创建新身份）
        
        Args:
            dense_vector: 稠密向量
            sparse_vector: 稀疏向量
            top_k: 返回数量
            
        Returns:
            相似人物列表
        """
        candidates = self.store.search(
            dense_vector=dense_vector,
            top_k=top_k
        )
        
        # 计算混合分数
        results = []
        for cand in candidates:
            dense_sim = cand['dense_similarity']
            sparse_sim = self._jaccard_similarity(
                sparse_vector,
                cand['sparse_vector']
            )
            
            total_score = self._calculate_total_score(dense_sim, sparse_sim)
            
            results.append({
                'person_uuid': cand['person_uuid'],
                'total_score': total_score,
                'dense_score': dense_sim,
                'sparse_score': sparse_sim,
                'attributes': cand['attributes'],
                'source': cand.get('source', {})
            })
        
        # 排序
        results.sort(key=lambda x: x['total_score'], reverse=True)
        
        return results


def create_matcher(
    store: ChromaStore,
    config: Optional[dict] = None
) -> IdentityMatcher:
    """从配置创建匹配器"""
    if config is None:
        return IdentityMatcher(store)
    
    return IdentityMatcher(
        store=store,
        threshold=config.get('threshold', 0.72),
        top_k=config.get('top_k', 5),
        weights=config.get('weights'),
        enable_face=config.get('enable_face', False)
    )
