#!/usr/bin/env python3
"""
CrossMedia-PID 视频人物匹配测试器

功能：
1. 读取测试视频，使用YOLOv8跟踪人物轨迹
2. 在视频中选择同一个人在不同位置（左侧/右侧）的两个瞬间截图
3. 将两个截图输入CrossMedia-PID系统，验证是否能正确匹配为同一人

使用方法：
    python test_video_person_matching.py <video_path> [options]

示例：
    python test_video_person_matching.py test_video.mp4 --output-dir ./test_output
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from ultralytics import YOLO

# 添加项目根目录到路径
project_root = Path(__file__).parent / "crossmedia_pid"
sys.path.insert(0, str(project_root))

from core.extractor import PersonExtractor, VisualOutput, BBox
from core.feature_vlm import FeatureExtractor, create_feature_extractor
from core.matcher import IdentityMatcher, MatchOutput
from core.vectorizer import DynamicVectorizer
from db.chroma_store import ChromaStore

console = Console()


@dataclass
class PersonTrack:
    """人物轨迹数据类"""
    track_id: int
    bbox_history: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    frame_history: List[int]
    confidence_history: List[float]
    best_frame: int
    best_bbox: Tuple[int, int, int, int]
    best_confidence: float
    best_quality: float


class VideoPersonTracker:
    """视频中的人物跟踪器"""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_bbox_size: int = 64,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_bbox_size = min_bbox_size
        
        # 加载YOLO模型
        console.print(f"[blue]Loading YOLO model: {model_path}[/blue]")
        self.model = YOLO(model_path)
        
        # 设置设备
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
                console.print("[green]Using MPS (Metal Performance Shaders)[/green]")
            else:
                device = 'cpu'
                console.print("[yellow]Using CPU[/yellow]")
        self.device = device
        
        # 跟踪器状态
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 0
        self.iou_threshold_tracking = 0.3  # 轨迹关联的IoU阈值
        
    def calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bbox_position(self, bbox: Tuple[int, int, int, int], frame_width: int) -> str:
        """
        判断边界框在画面中的位置
        将画面分为左、中、右三个区域
        """
        cx, _ = self.get_bbox_center(bbox)
        if cx < frame_width * 0.33:
            return "left"
        elif cx > frame_width * 0.67:
            return "right"
        else:
            return "center"
    
    def calculate_quality_score(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ) -> float:
        """计算检测质量评分"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        frame_area = h * w
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # 大小评分 (0.3)
        size_ratio = bbox_area / frame_area
        size_score = min(size_ratio * 5, 1.0)
        
        # 清晰度评分 (0.3)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            clarity_score = 0.0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.LAPLACIAN).var()
            clarity_score = min(laplacian_var / 100, 1.0)
        
        # 位置评分 (0.1)
        cx, cy = self.get_bbox_center(bbox)
        center_dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
        max_dist = ((w/2)**2 + (h/2)**2) ** 0.5
        position_score = 1.0 - (center_dist / max_dist)
        
        # 综合评分
        quality_score = (
            0.3 * confidence +
            0.3 * size_score +
            0.3 * clarity_score +
            0.1 * position_score
        )
        
        return quality_score
    
    def update_tracks(
        self,
        frame: np.ndarray,
        detections: List[Tuple[Tuple[int, int, int, int], float]],
        frame_number: int
    ):
        """更新人物轨迹"""
        h, w = frame.shape[:2]
        
        # 计算每个检测的质量分数
        scored_detections = []
        for bbox, conf in detections:
            quality = self.calculate_quality_score(frame, bbox, conf)
            scored_detections.append((bbox, conf, quality))
        
        # 简单的IoU关联跟踪
        matched_tracks = set()
        matched_detections = set()
        
        # 按质量排序，优先匹配高质量检测
        scored_detections.sort(key=lambda x: x[2], reverse=True)
        
        for det_idx, (bbox, conf, quality) in enumerate(scored_detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # 计算与轨迹最新位置的IoU
                last_bbox = track.bbox_history[-1]
                iou = self.calculate_iou(bbox, last_bbox)
                
                if iou > best_iou and iou > self.iou_threshold_tracking:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # 更新现有轨迹
                track = self.tracks[best_track_id]
                track.bbox_history.append(bbox)
                track.frame_history.append(frame_number)
                track.confidence_history.append(conf)
                
                # 更新最佳帧
                if quality > track.best_quality:
                    track.best_quality = quality
                    track.best_frame = frame_number
                    track.best_bbox = bbox
                    track.best_confidence = conf
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            else:
                # 创建新轨迹
                new_track = PersonTrack(
                    track_id=self.next_track_id,
                    bbox_history=[bbox],
                    frame_history=[frame_number],
                    confidence_history=[conf],
                    best_frame=frame_number,
                    best_bbox=bbox,
                    best_confidence=conf,
                    best_quality=quality
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
                matched_detections.add(det_idx)
    
    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 5,
        max_frames: Optional[int] = None,
        progress_callback=None
    ) -> Dict[int, PersonTrack]:
        """
        处理视频，提取人物轨迹
        
        Args:
            video_path: 视频路径
            sample_interval: 采样间隔（帧）
            max_frames: 最大处理帧数
            progress_callback: 进度回调函数
            
        Returns:
            人物轨迹字典
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        console.print(f"\n[bold cyan]Video Info:[/bold cyan]")
        console.print(f"  Resolution: {width}x{height}")
        console.print(f"  FPS: {fps:.2f}")
        console.print(f"  Total frames: {total_frames}")
        console.print(f"  Sample interval: every {sample_interval} frames")
        
        frame_count = 0
        processed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing video...", total=total_frames // sample_interval if max_frames is None else max_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 按间隔采样
                if frame_count % sample_interval != 0:
                    continue
                
                # YOLO检测
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=[0],  # person class only
                    device=self.device,
                    verbose=False
                )
                
                # 提取检测结果
                detections = []
                for result in results:
                    if result.boxes is None:
                        continue
                    
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # 过滤太小的框
                        if (x2 - x1) < self.min_bbox_size or (y2 - y1) < self.min_bbox_size:
                            continue
                        
                        detections.append(((x1, y1, x2, y2), float(conf)))
                
                # 更新轨迹
                self.update_tracks(frame, detections, frame_count)
                
                processed_count += 1
                progress.update(task, advance=1)
                
                if max_frames and processed_count >= max_frames:
                    break
        
        cap.release()
        
        console.print(f"\n[green]Video processing complete![/green]")
        console.print(f"  Processed frames: {processed_count}")
        console.print(f"  Detected tracks: {len(self.tracks)}")
        
        return self.tracks
    
    def find_person_at_positions(
        self,
        video_path: Path,
        target_track_id: int,
        positions: List[str] = ["left", "right"]
    ) -> Dict[str, Tuple[int, Tuple[int, int, int, int]]]:
        """
        查找指定人物在特定位置的帧
        
        Args:
            video_path: 视频路径
            target_track_id: 目标轨迹ID
            positions: 需要查找的位置列表 ["left", "right", "center"]
            
        Returns:
            位置到(帧号, bbox)的映射
        """
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        track = self.tracks.get(target_track_id)
        if not track:
            cap.release()
            return {}
        
        result = {}
        found_positions = set()
        
        # 遍历轨迹历史，查找指定位置
        for frame_num, bbox in zip(track.frame_history, track.bbox_history):
            pos = self.get_bbox_position(bbox, width)
            
            if pos in positions and pos not in found_positions:
                result[pos] = (frame_num, bbox)
                found_positions.add(pos)
                
                if len(found_positions) == len(positions):
                    break
        
        cap.release()
        return result
    
    def extract_frame_crop(
        self,
        video_path: Path,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> np.ndarray:
        """
        从视频中提取指定帧的裁剪区域
        
        Args:
            video_path: 视频路径
            frame_number: 帧号
            bbox: 边界框 (x1, y1, x2, y2)
            padding: 边距比例
            
        Returns:
            裁剪后的图像
        """
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read frame {frame_number}")
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # 添加边距
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        return crop


class CrossMediaPIDTester:
    """CrossMedia-PID测试器"""
    
    def __init__(self, config_path: str = "crossmedia_pid/configs/config.yaml"):
        """初始化测试器"""
        self.config = self._load_config(config_path)
        
        # 初始化各模块
        console.print("\n[bold blue]Initializing CrossMedia-PID...[/bold blue]")
        
        # A模块：视觉提取
        yolo_config = self.config.get('models', {}).get('yolo', {})
        self.extractor = PersonExtractor(
            model_path=yolo_config.get('model_path', 'yolov8n.pt'),
            conf_threshold=yolo_config.get('conf_threshold', 0.5),
            iou_threshold=yolo_config.get('iou_threshold', 0.45),
            min_bbox_size=self.config.get('features', {}).get('min_bbox_size', 64)
        )
        
        # B模块：特征提取
        vlm_config = self.config.get('models', {}).get('vlm', {})
        
        # 检查API密钥
        provider = vlm_config.get('provider', 'cloud')
        if provider == 'cloud':
            api_key = vlm_config.get('api_key', '')
            if not api_key:
                api_key = os.getenv('VLM_API_KEY', '')
            vlm_config['api_key'] = api_key
        elif provider == 'aliyun':
            api_key = vlm_config.get('api_key', '')
            if not api_key:
                api_key = os.getenv('DASHSCOPE_API_KEY', '')
            vlm_config['api_key'] = api_key
        
        self.feature_extractor = create_feature_extractor(vlm_config)
        
        # C模块：向量化
        embedding_config = self.config.get('models', {}).get('embedding', {})
        registry_config = self.config.get('registry', {})
        self.vectorizer = DynamicVectorizer(
            dense_model_name=embedding_config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
            max_length=embedding_config.get('max_length', 512),
            registry_path=registry_config.get('persist_path', './attribute_registry.json')
        )
        
        # D模块：匹配
        chroma_config = self.config.get('database', {}).get('chroma', {})
        self.store = ChromaStore(
            persist_directory=chroma_config.get('persist_directory', './chroma_db'),
            collection_name=chroma_config.get('collection_name', 'person_embeddings'),
            distance_fn=chroma_config.get('distance_fn', 'cosine')
        )
        
        matching_config = self.config.get('matching', {})
        self.matcher = IdentityMatcher(
            store=self.store,
            threshold=matching_config.get('threshold', 0.72),
            top_k=matching_config.get('top_k', 5),
            weights=matching_config.get('weights'),
            enable_face=False
        )
        
        console.print("[bold green]System initialized successfully![/bold green]")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        import yaml
        
        config_file = Path(config_path)
        if not config_file.exists():
            console.print(f"[yellow]Warning: Config file not found: {config_path}[/yellow]")
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 处理环境变量替换
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default_val = ""
                if ':' in env_var:
                    env_var, default_val = env_var.split(':', 1)
                return os.getenv(env_var, default_val)
            return obj
        
        return replace_env_vars(config)
    
    def process_image(self, image: np.ndarray, source_name: str) -> Optional[Dict]:
        """
        处理单张图片
        
        Args:
            image: numpy数组格式的图片
            source_name: 来源名称（用于日志）
            
        Returns:
            处理结果字典
        """
        console.print(f"\n[bold]Processing:[/bold] {source_name}")
        
        start_time = time.time()
        
        # Step 1: 视觉提取 (A模块)
        console.print("  [yellow]Step 1/4:[/yellow] Visual extraction...", end=" ")
        
        # 临时保存图片用于extractor
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)
        
        try:
            visual_output = self.extractor.extract(tmp_path, return_best_only=True)
        finally:
            os.unlink(tmp_path)
        
        if visual_output is None:
            console.print("[red]FAILED - No person detected[/red]")
            return None
        
        console.print(f"[green]OK[/green] (quality={visual_output.quality_score:.2f})")
        
        # Step 2: 特征提取 (B模块)
        console.print("  [yellow]Step 2/4:[/yellow] Feature extraction...", end=" ")
        feature_output = self.feature_extractor.extract(visual_output.crop_image)
        
        if not feature_output.is_valid:
            console.print(f"[red]FAILED - {feature_output.raw_response[:50]}...[/red]")
            return None
        
        console.print(f"[green]OK[/green] ({len(feature_output.attributes)} attributes)")
        
        # Step 3: 向量化 (C模块)
        console.print("  [yellow]Step 3/4:[/yellow] Vectorization...", end=" ")
        vector_output = self.vectorizer.vectorize(
            feature_output.attributes,
            source_meta={'source_path': source_name, 'quality_score': visual_output.quality_score}
        )
        console.print("[green]OK[/green]")
        
        # Step 4: 身份匹配 (D模块)
        console.print("  [yellow]Step 4/4:[/yellow] Identity matching...", end=" ")
        match_output = self.matcher.match(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            query_attributes=feature_output.attributes
        )
        
        if match_output.is_new_identity:
            console.print(f"[cyan]NEW IDENTITY[/cyan] ({match_output.person_uuid})")
        else:
            console.print(f"[green]MATCHED[/green] ({match_output.person_uuid}, score={match_output.match_score:.3f})")
        
        elapsed = time.time() - start_time
        console.print(f"  [dim]Total time: {elapsed:.2f}s[/dim]")
        
        return {
            'source_name': source_name,
            'person_uuid': match_output.person_uuid,
            'is_new': match_output.is_new_identity,
            'match_score': match_output.match_score,
            'attributes': feature_output.attributes,
            'quality_score': visual_output.quality_score,
            'elapsed_time': elapsed,
            'dense_vector': vector_output.dense_vector,
            'sparse_vector': vector_output.sparse_vector
        }
    
    def add_to_database(self, result: Dict):
        """将结果添加到数据库"""
        self.matcher.add_identity(
            person_uuid=result['person_uuid'],
            dense_vector=result['dense_vector'],
            sparse_vector=result['sparse_vector'],
            attributes=result['attributes'],
            source_meta={
                'source_path': result['source_name'],
                'quality_score': result['quality_score']
            }
        )


def run_test(
    video_path: Path,
    output_dir: Path,
    sample_interval: int = 5,
    max_frames: Optional[int] = None,
    save_crops: bool = True
) -> Dict:
    """
    运行完整的视频人物匹配测试
    
    Args:
        video_path: 测试视频路径
        output_dir: 输出目录
        sample_interval: 采样间隔
        max_frames: 最大处理帧数
        save_crops: 是否保存裁剪图片
        
    Returns:
        测试结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]CrossMedia-PID Video Person Matching Test[/bold cyan]")
    console.print("=" * 60)
    
    # Step 1: 视频跟踪
    console.print("\n[bold]Step 1: Video Tracking[/bold]")
    tracker = VideoPersonTracker()
    tracks = tracker.process_video(video_path, sample_interval, max_frames)
    
    if not tracks:
        console.print("[red]No person tracks detected![/red]")
        return {'success': False, 'error': 'No tracks detected'}
    
    # 选择最佳轨迹（持续时间最长或质量最高）
    best_track_id = max(tracks.keys(), key=lambda tid: len(tracks[tid].frame_history))
    best_track = tracks[best_track_id]
    
    console.print(f"\n[green]Selected track #{best_track_id}:[/green]")
    console.print(f"  Frames tracked: {len(best_track.frame_history)}")
    console.print(f"  Best quality: {best_track.best_quality:.3f}")
    
    # Step 2: 查找左右位置的帧
    console.print("\n[bold]Step 2: Finding positions (left/right)[/bold]")
    positions = tracker.find_person_at_positions(video_path, best_track_id, ["left", "right"])
    
    if len(positions) < 2:
        console.print(f"[yellow]Warning: Only found {len(positions)} positions[/yellow]")
        # 如果没有找到左右两个位置，使用轨迹中的第一帧和最后一帧
        if len(best_track.frame_history) >= 2:
            first_frame = best_track.frame_history[0]
            last_frame = best_track.frame_history[-1]
            first_bbox = best_track.bbox_history[0]
            last_bbox = best_track.bbox_history[-1]
            
            positions = {
                "first": (first_frame, first_bbox),
                "last": (last_frame, last_bbox)
            }
            console.print("[yellow]Using first/last frames instead[/yellow]")
        else:
            return {'success': False, 'error': 'Insufficient frames for testing'}
    
    # 显示找到的帧
    for pos, (frame_num, bbox) in positions.items():
        console.print(f"  [cyan]{pos}:[/cyan] Frame {frame_num}, bbox={bbox}")
    
    # Step 3: 提取裁剪图片
    console.print("\n[bold]Step 3: Extracting crops[/bold]")
    crops = {}
    for pos, (frame_num, bbox) in positions.items():
        crop = tracker.extract_frame_crop(video_path, frame_num, bbox)
        crops[pos] = crop
        
        if save_crops:
            crop_path = output_dir / f"crop_{pos}_frame{frame_num}.jpg"
            cv2.imwrite(str(crop_path), crop)
            console.print(f"  [green]Saved:[/green] {crop_path}")
    
    # Step 4: 使用CrossMedia-PID处理
    console.print("\n[bold]Step 4: CrossMedia-PID Processing[/bold]")
    tester = CrossMediaPIDTester()
    
    results = []
    for pos, crop in crops.items():
        result = tester.process_image(crop, f"{pos}_frame{positions[pos][0]}")
        if result:
            results.append(result)
            # 第一个添加到数据库，第二个用于匹配测试
            if len(results) == 1:
                tester.add_to_database(result)
                console.print(f"  [dim]Added to database as reference[/dim]")
    
    if len(results) < 2:
        return {'success': False, 'error': 'Failed to process both images'}
    
    # Step 5: 分析结果
    console.print("\n[bold]Step 5: Analysis[/bold]")
    
    result1, result2 = results[0], results[1]
    
    # 判断是否匹配到同一个人
    same_person = result1['person_uuid'] == result2['person_uuid']
    
    # 创建结果表格
    table = Table(title="Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Image 1", style="green")
    table.add_column("Image 2", style="green")
    
    table.add_row("Position", list(crops.keys())[0], list(crops.keys())[1])
    table.add_row("Person UUID", result1['person_uuid'], result2['person_uuid'])
    table.add_row("Is New Identity", str(result1['is_new']), str(result2['is_new']))
    table.add_row("Match Score", "N/A (reference)", f"{result2['match_score']:.3f}")
    table.add_row("Quality Score", f"{result1['quality_score']:.3f}", f"{result2['quality_score']:.3f}")
    table.add_row("Processing Time", f"{result1['elapsed_time']:.2f}s", f"{result2['elapsed_time']:.2f}s")
    
    console.print(table)
    
    # 测试结论
    console.print("\n" + "=" * 60)
    if same_person:
        console.print("[bold green]✓ TEST PASSED: Same person correctly identified![/bold green]")
    else:
        console.print("[bold red]✗ TEST FAILED: Images identified as different persons[/bold red]")
    console.print("=" * 60)
    
    # 保存详细结果
    test_result = {
        'success': True,
        'test_passed': same_person,
        'video_path': str(video_path),
        'track_id': best_track_id,
        'positions': {k: {'frame': v[0], 'bbox': v[1]} for k, v in positions.items()},
        'image1': {
            'position': list(positions.keys())[0],
            'person_uuid': result1['person_uuid'],
            'is_new': result1['is_new'],
            'quality_score': result1['quality_score'],
            'attributes': result1['attributes']
        },
        'image2': {
            'position': list(positions.keys())[1],
            'person_uuid': result2['person_uuid'],
            'is_new': result2['is_new'],
            'match_score': result2['match_score'],
            'quality_score': result2['quality_score'],
            'attributes': result2['attributes']
        },
        'same_person': same_person
    }
    
    result_path = output_dir / "test_result.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Detailed result saved to: {result_path}[/dim]")
    
    return test_result


def main():
    parser = argparse.ArgumentParser(
        description="CrossMedia-PID Video Person Matching Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_video_person_matching.py test_video.mp4
    python test_video_person_matching.py test_video.mp4 --output-dir ./test_output --sample-interval 10
        """
    )
    
    parser.add_argument('video_path', type=str, help='Path to test video (1080P recommended)')
    parser.add_argument('--output-dir', '-o', type=str, default='./test_output',
                        help='Output directory for test results (default: ./test_output)')
    parser.add_argument('--sample-interval', '-s', type=int, default=5,
                        help='Frame sampling interval (default: 5)')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--no-save-crops', action='store_true',
                        help='Do not save cropped images')
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        console.print(f"[red]Error: Video file not found: {video_path}[/red]")
        sys.exit(1)
    
    # 运行测试
    result = run_test(
        video_path=video_path,
        output_dir=Path(args.output_dir),
        sample_interval=args.sample_interval,
        max_frames=args.max_frames,
        save_crops=not args.no_save_crops
    )
    
    # 退出码
    sys.exit(0 if result.get('test_passed', False) else 1)


if __name__ == '__main__':
    main()
