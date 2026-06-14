"""
Module A: Visual Extraction
人体检测、最佳帧筛选、ROI裁剪
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from crossmedia_pid.config import project_path

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """边界框数据类"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    """检测结果数据类"""
    bbox: BBox
    confidence: float
    class_id: int
    

@dataclass
class VisualOutput:
    """A模块输出"""
    crop_image: np.ndarray
    bbox: BBox
    quality_score: float
    source_path: str
    detection_confidence: float


class PersonExtractor:
    """人体提取器"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_bbox_size: int = 64,
        device: Optional[str] = None
    ):
        """
        初始化人体提取器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IoU阈值
            min_bbox_size: 最小边界框尺寸
            device: 推理设备 ('mps', 'cpu', None for auto)
        """
        if model_path is None:
            model_path = str(project_path("models", "yolov8n.pt"))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_bbox_size = min_bbox_size
        
        # 加载YOLO模型
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
        # 设置设备
        if device is None:
            # M1 Mac优先使用MPS
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
                logger.info("Using MPS (Metal Performance Shaders) for YOLO")
            else:
                device = 'cpu'
                logger.info("Using CPU for YOLO")
        self.device = device
        
    def detect(self, image: Union[np.ndarray, str, Path]) -> List[Detection]:
        """
        检测图片中的人体
        
        Args:
            image: 输入图片 (numpy数组或路径)
            
        Returns:
            检测到的目标列表
        """
        # 执行推理
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # 只检测person类
            device=self.device,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                bbox = BBox(x1, y1, x2, y2)
                
                # 过滤太小的框
                if bbox.width < self.min_bbox_size or bbox.height < self.min_bbox_size:
                    continue
                
                detections.append(Detection(
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls_id)
                ))
        
        return detections
    
    def calculate_quality_score(
        self,
        image: np.ndarray,
        bbox: BBox,
        detection_conf: float
    ) -> float:
        """
        计算裁剪区域的质量评分
        
        综合因素：
        - 检测置信度
        - 边界框大小（相对于图片）
        - 图像清晰度（拉普拉斯方差）
        
        Args:
            image: 原始图片
            bbox: 边界框
            detection_conf: 检测置信度
            
        Returns:
            质量评分 (0-1)
        """
        h, w = image.shape[:2]
        image_area = h * w
        
        # 大小评分 (0.3)
        size_ratio = bbox.area / image_area
        size_score = min(size_ratio * 5, 1.0)  # 放大5倍，但不超过1
        
        # 清晰度评分 (0.3)
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        if crop.size == 0:
            clarity_score = 0.0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # 归一化到0-1，经验阈值100
            clarity_score = min(laplacian_var / 100, 1.0)
        
        # 位置评分 (0.1) - 中心位置更好
        cx, cy = bbox.center
        center_dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
        max_dist = ((w/2)**2 + (h/2)**2) ** 0.5
        position_score = 1.0 - (center_dist / max_dist)
        
        # 综合评分
        quality_score = (
            0.3 * detection_conf +
            0.3 * size_score +
            0.3 * clarity_score +
            0.1 * position_score
        )
        
        return quality_score
    
    def extract(
        self,
        image_path: Union[str, Path],
        return_best_only: bool = True,
        min_quality: float = 0.3
    ) -> Union[Optional[VisualOutput], List[VisualOutput]]:
        """
        从图片中提取人体
        
        Args:
            image_path: 图片路径
            return_best_only: 是否只返回最佳结果
            min_quality: 最小质量阈值
            
        Returns:
            VisualOutput或列表
        """
        image_path = Path(image_path)
        
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None if return_best_only else []
        
        # 检测人体
        detections = self.detect(image)
        
        if not detections:
            logger.warning(f"No person detected in {image_path}")
            return None if return_best_only else []
        
        # 计算质量并提取
        outputs = []
        for det in detections:
            quality = self.calculate_quality_score(
                image, det.bbox, det.confidence
            )
            
            if quality < min_quality:
                continue
            
            # 裁剪ROI
            crop = image[det.bbox.y1:det.bbox.y2, det.bbox.x1:det.bbox.x2]
            
            outputs.append(VisualOutput(
                crop_image=crop,
                bbox=det.bbox,
                quality_score=quality,
                source_path=str(image_path),
                detection_confidence=det.confidence
            ))
        
        # 按质量排序
        outputs.sort(key=lambda x: x.quality_score, reverse=True)
        
        if return_best_only:
            return outputs[0] if outputs else None
        return outputs
    
    def extract_from_video(
        self,
        video_path: Union[str, Path],
        sample_interval: int = 5,
        max_frames: Optional[int] = None
    ) -> List[VisualOutput]:
        """
        从视频中提取人体（最佳帧筛选）
        
        Args:
            video_path: 视频路径
            sample_interval: 采样间隔（帧）
            max_frames: 最大处理帧数
            
        Returns:
            提取结果列表
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        all_outputs = []
        frame_count = 0
        processed_count = 0
        
        logger.info(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 按间隔采样
            if frame_count % sample_interval != 0:
                continue
            
            # 检测
            detections = self.detect(frame)
            
            for det in detections:
                quality = self.calculate_quality_score(
                    frame, det.bbox, det.confidence
                )
                
                crop = frame[det.bbox.y1:det.bbox.y2, det.bbox.x1:det.bbox.x2]
                
                all_outputs.append(VisualOutput(
                    crop_image=crop,
                    bbox=det.bbox,
                    quality_score=quality,
                    source_path=f"{video_path}#frame_{frame_count}",
                    detection_confidence=det.confidence
                ))
            
            processed_count += 1
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
        
        # 按质量排序，去重（同一轨迹只保留最佳）
        all_outputs.sort(key=lambda x: x.quality_score, reverse=True)
        
        logger.info(f"Video processed: {frame_count} frames, {len(all_outputs)} detections")
        
        return all_outputs


# 便捷函数
def create_extractor(config: Optional[dict] = None) -> PersonExtractor:
    """从配置创建提取器"""
    if config is None:
        return PersonExtractor()
    
    return PersonExtractor(
        model_path=config.get('model_path', str(project_path("models", "yolov8n.pt"))),
        conf_threshold=config.get('conf_threshold', 0.5),
        iou_threshold=config.get('iou_threshold', 0.45),
        min_bbox_size=config.get('min_bbox_size', 64)
    )
