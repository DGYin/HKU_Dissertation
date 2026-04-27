"""
FastAPI Routes (Phase 1基础框架)
为Phase 2的Web UI做准备
"""

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1")


class ExtractResponse(BaseModel):
    """提取响应"""
    person_uuid: str
    features: dict
    status: str
    message: Optional[str] = None


class SearchResult(BaseModel):
    """搜索结果"""
    uuid: str
    score: float
    traits: List[str]
    trajectories: List[dict]


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[SearchResult]
    total: int


@router.post("/extract", response_model=ExtractResponse)
async def extract_features(
    file: UploadFile = File(...),
    add_to_db: bool = Form(True)
):
    """
    从图片提取特征
    
    - **file**: 上传的图片文件
    - **add_to_db**: 是否添加到数据库
    """
    # TODO: Phase 2实现
    return ExtractResponse(
        person_uuid="p_placeholder",
        features={},
        status="not_implemented",
        message="This endpoint will be implemented in Phase 2"
    )


@router.get("/search", response_model=SearchResponse)
async def search_persons(
    q: Optional[str] = None,
    image_url: Optional[str] = None,
    topk: int = 5,
    threshold: float = 0.72
):
    """
    搜索人物
    
    - **q**: 文本查询（自然语言描述）
    - **image_url**: 图片URL（以图搜图）
    - **topk**: 返回结果数量
    - **threshold**: 相似度阈值
    """
    # TODO: Phase 2实现
    return SearchResponse(results=[], total=0)


@router.get("/person/{person_uuid}")
async def get_person(person_uuid: str):
    """获取人物详情"""
    # TODO: Phase 2实现
    raise HTTPException(status_code=404, detail="Not implemented in Phase 1")


@router.get("/stats")
async def get_stats():
    """获取系统统计"""
    # TODO: Phase 2实现
    return {
        "total_persons": 0,
        "total_records": 0,
        "registry_attributes": 0
    }
