#!/usr/bin/env python3
"""
CrossMedia-PID 视频人物匹配测试器 - Web GUI版本

功能：
1. 拖拽上传视频文件
2. 使用YOLOv8检测并跟踪人物轨迹
3. 可视化展示检测到的人物及其位置分布
4. 手动选择待测试的人物轨迹
5. 自动提取左右位置截图并测试CrossMedia-PID匹配

使用方法：
    python video_test_webgui.py
    
    然后打开浏览器访问: http://localhost:8501
"""

import base64
import io
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置Streamlit环境变量，禁用邮件收集
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# 添加项目根目录到路径
project_root = Path(__file__).parent / "crossmedia_pid"
sys.path.insert(0, str(project_root))

from core.extractor import PersonExtractor
from core.feature_vlm import create_feature_extractor
from core.matcher import IdentityMatcher
from core.vectorizer import DynamicVectorizer
from db.chroma_store import ChromaStore

# 页面配置
st.set_page_config(
    page_title="CrossMedia-PID 视频测试器",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .track-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
    }
    .track-selected {
        border-left: 4px solid #27ae60;
        background-color: #e8f5e9;
    }
    .metric-box {
        background-color: #f1f3f4;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .status-success {
        color: #27ae60;
        font-weight: bold;
    }
    .status-fail {
        color: #e74c3c;
        font-weight: bold;
    }
    .crop-image {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class PersonTrack:
    """人物轨迹数据类"""
    track_id: int
    bbox_history: List[Tuple[int, int, int, int]]
    frame_history: List[int]
    confidence_history: List[float]
    position_history: List[str]
    best_frame: int
    best_bbox: Tuple[int, int, int, int]
    best_confidence: float
    best_quality: float
    frame_width: int
    frame_height: int


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
        self.model = YOLO(model_path)
        
        # 设置设备
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # 跟踪器状态
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 0
        self.iou_threshold_tracking = 0.3
        self.frame_width = 0
        self.frame_height = 0
        
    def calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bbox_position(self, bbox: Tuple[int, int, int, int]) -> str:
        """判断边界框在画面中的位置"""
        cx, _ = self.get_bbox_center(bbox)
        if cx < self.frame_width * 0.33:
            return "left"
        elif cx > self.frame_width * 0.67:
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
        
        size_ratio = bbox_area / frame_area
        size_score = min(size_ratio * 5, 1.0)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            clarity_score = 0.0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = min(laplacian_var / 100, 1.0)
        
        cx, cy = self.get_bbox_center(bbox)
        center_dist = ((cx - w/2)**2 + (cy - h/2)**2) ** 0.5
        max_dist = ((w/2)**2 + (h/2)**2) ** 0.5
        position_score = 1.0 - (center_dist / max_dist)
        
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
        scored_detections = []
        for bbox, conf in detections:
            quality = self.calculate_quality_score(frame, bbox, conf)
            scored_detections.append((bbox, conf, quality))
        
        matched_tracks = set()
        scored_detections.sort(key=lambda x: x[2], reverse=True)
        
        for det_idx, (bbox, conf, quality) in enumerate(scored_detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                last_bbox = track.bbox_history[-1]
                iou = self.calculate_iou(bbox, last_bbox)
                
                if iou > best_iou and iou > self.iou_threshold_tracking:
                    best_iou = iou
                    best_track_id = track_id
            
            position = self.get_bbox_position(bbox)
            
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                track.bbox_history.append(bbox)
                track.frame_history.append(frame_number)
                track.confidence_history.append(conf)
                track.position_history.append(position)
                
                if quality > track.best_quality:
                    track.best_quality = quality
                    track.best_frame = frame_number
                    track.best_bbox = bbox
                    track.best_confidence = conf
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            else:
                new_track = PersonTrack(
                    track_id=self.next_track_id,
                    bbox_history=[bbox],
                    frame_history=[frame_number],
                    confidence_history=[conf],
                    position_history=[position],
                    best_frame=frame_number,
                    best_bbox=bbox,
                    best_confidence=conf,
                    best_quality=quality,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
    
    def process_video(
        self,
        video_path: Path,
        sample_interval: int = 5,
        max_frames: Optional[int] = None,
        progress_bar=None
    ) -> Dict[int, PersonTrack]:
        """处理视频，提取人物轨迹"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_count = 0
        processed_count = 0
        total_to_process = total_frames // sample_interval if max_frames is None else max_frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_interval != 0:
                continue
            
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[0],
                device=self.device,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    
                    if (x2 - x1) < self.min_bbox_size or (y2 - y1) < self.min_bbox_size:
                        continue
                    
                    detections.append(((x1, y1, x2, y2), float(conf)))
            
            self.update_tracks(frame, detections, frame_count)
            
            processed_count += 1
            if progress_bar:
                progress_bar.progress(min(processed_count / total_to_process, 1.0))
            
            if max_frames and processed_count >= max_frames:
                break
        
        cap.release()
        return self.tracks
    
    def get_track_preview_image(self, video_path: Path, track: PersonTrack) -> np.ndarray:
        """获取轨迹预览图（最佳帧）"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, track.best_frame - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # 绘制边界框
        x1, y1, x2, y2 = track.best_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def extract_frame_crop(
        self,
        video_path: Path,
        frame_number: int,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> np.ndarray:
        """提取帧裁剪区域"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"无法读取帧 {frame_number}")
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        crop = frame[y1:y2, x1:x2]
        return crop
    
    def find_frames_at_positions(
        self,
        video_path: Path,
        track_id: int,
        target_positions: List[str]
    ) -> Dict[str, Tuple[int, np.ndarray]]:
        """查找指定位置的帧"""
        track = self.tracks.get(track_id)
        if not track:
            return {}
        
        result = {}
        found_positions = set()
        
        for frame_num, bbox, pos in zip(track.frame_history, track.bbox_history, track.position_history):
            if pos in target_positions and pos not in found_positions:
                crop = self.extract_frame_crop(video_path, frame_num, bbox)
                result[pos] = (frame_num, crop)
                found_positions.add(pos)
                
                if len(found_positions) == len(target_positions):
                    break
        
        return result


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """将numpy数组转换为PIL图像"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_position_distribution(positions: List[str]) -> Dict[str, int]:
    """统计位置分布"""
    dist = {"left": 0, "center": 0, "right": 0}
    for pos in positions:
        if pos in dist:
            dist[pos] += 1
    return dist


def init_crossmedia_pid():
    """初始化CrossMedia-PID系统"""
    import yaml
    
    config_path = Path("crossmedia_pid/configs/config.yaml")
    if not config_path.exists():
        st.error("配置文件不存在: crossmedia_pid/configs/config.yaml")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 处理环境变量
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
    
    config = replace_env_vars(config)
    
    # 初始化各模块
    yolo_config = config.get('models', {}).get('yolo', {})
    extractor = PersonExtractor(
        model_path=yolo_config.get('model_path', 'yolov8n.pt'),
        conf_threshold=yolo_config.get('conf_threshold', 0.5),
        iou_threshold=yolo_config.get('iou_threshold', 0.45),
        min_bbox_size=config.get('features', {}).get('min_bbox_size', 64)
    )
    
    vlm_config = config.get('models', {}).get('vlm', {})
    provider = vlm_config.get('provider', 'cloud')
    if provider == 'cloud':
        api_key = vlm_config.get('api_key', '') or os.getenv('VLM_API_KEY', '')
        vlm_config['api_key'] = api_key
    elif provider == 'aliyun':
        api_key = vlm_config.get('api_key', '') or os.getenv('DASHSCOPE_API_KEY', '')
        vlm_config['api_key'] = api_key
    
    feature_extractor = create_feature_extractor(vlm_config)
    
    embedding_config = config.get('models', {}).get('embedding', {})
    registry_config = config.get('registry', {})
    vectorizer = DynamicVectorizer(
        dense_model_name=embedding_config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
        max_length=embedding_config.get('max_length', 512),
        registry_path=registry_config.get('persist_path', './attribute_registry.json')
    )
    
    chroma_config = config.get('database', {}).get('chroma', {})
    store = ChromaStore(
        persist_directory=chroma_config.get('persist_directory', './chroma_db'),
        collection_name=chroma_config.get('collection_name', 'person_embeddings'),
        distance_fn=chroma_config.get('distance_fn', 'cosine')
    )
    
    matching_config = config.get('matching', {})
    matcher = IdentityMatcher(
        store=store,
        threshold=matching_config.get('threshold', 0.72),
        top_k=matching_config.get('top_k', 5),
        weights=matching_config.get('weights'),
        enable_face=False
    )
    
    return {
        'extractor': extractor,
        'feature_extractor': feature_extractor,
        'vectorizer': vectorizer,
        'store': store,
        'matcher': matcher
    }


def process_image_with_pid(pid_system: dict, image: np.ndarray, source_name: str) -> Optional[dict]:
    """使用CrossMedia-PID处理单张图片"""
    import tempfile
    
    start_time = time.time()
    
    # 临时保存图片
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image)
    
    try:
        # 视觉提取
        visual_output = pid_system['extractor'].extract(tmp_path, return_best_only=True)
        if visual_output is None:
            return None
        
        # 特征提取
        feature_output = pid_system['feature_extractor'].extract(visual_output.crop_image)
        if not feature_output.is_valid:
            return None
        
        # 向量化
        vector_output = pid_system['vectorizer'].vectorize(
            feature_output.attributes,
            source_meta={'source_path': source_name, 'quality_score': visual_output.quality_score}
        )
        
        # 身份匹配
        match_output = pid_system['matcher'].match(
            dense_vector=vector_output.dense_vector,
            sparse_vector=vector_output.sparse_vector,
            query_attributes=feature_output.attributes
        )
        
        elapsed = time.time() - start_time
        
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
    finally:
        os.unlink(tmp_path)


def main():
    # 页面标题
    st.markdown('<div class="main-header">🔍 CrossMedia-PID 视频人物匹配测试器</div>', unsafe_allow_html=True)
    
    st.markdown("""
    本工具用于测试CrossMedia-PID系统的人物识别能力：
    1. 上传监控视频
    2. 系统自动检测并跟踪人物轨迹
    3. 选择待测试的人物
    4. 系统自动提取左右位置的截图
    5. 测试CrossMedia-PID是否能正确匹配为同一人
    """)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置参数")
        
        sample_interval = st.slider("采样间隔 (帧)", 1, 30, 5, 
                                    help="每N帧处理一次，值越大处理越快但可能漏检")
        max_frames = st.number_input("最大处理帧数", min_value=0, max_value=10000, value=0,
                                     help="0表示不限制")
        max_frames = None if max_frames == 0 else max_frames
        
        st.divider()
        st.info("💡 提示：处理完成后，请在主界面选择要测试的人物轨迹")
    
    # 初始化session state
    if 'tracks' not in st.session_state:
        st.session_state.tracks = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'selected_track_id' not in st.session_state:
        st.session_state.selected_track_id = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'pid_system' not in st.session_state:
        st.session_state.pid_system = None
    
    # 步骤1: 视频上传
    st.markdown('<div class="section-header">📹 步骤1: 上传测试视频</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "拖拽视频文件到此处，或点击选择文件",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持MP4、AVI、MOV、MKV格式，建议使用1080P视频"
    )
    
    if uploaded_file is not None:
        # 保存上传的视频
        video_path = Path(tempfile.gettempdir()) / f"uploaded_{uploaded_file.name}"
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_path = video_path
        
        # 显示视频信息
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{width}×{height}</div><div class="metric-label">分辨率</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{fps:.1f}</div><div class="metric-label">帧率</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{frame_count}</div><div class="metric-label">总帧数</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{duration:.1f}s</div><div class="metric-label">时长</div></div>', unsafe_allow_html=True)
        
        # 处理按钮
        if st.button("🚀 开始分析视频", type="primary", use_container_width=True):
            st.session_state.tracks = None
            st.session_state.selected_track_id = None
            st.session_state.test_results = None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("正在分析视频中的人物轨迹..."):
                tracker = VideoPersonTracker()
                status_text.text("初始化YOLO模型...")
                
                tracks = tracker.process_video(
                    video_path,
                    sample_interval=sample_interval,
                    max_frames=max_frames,
                    progress_bar=progress_bar
                )
                
                st.session_state.tracks = tracks
                progress_bar.empty()
                status_text.empty()
            
            st.success(f"✅ 分析完成！检测到 {len(tracks)} 个人物轨迹")
            st.rerun()
    
    # 步骤2: 显示检测结果并选择
    if st.session_state.tracks:
        st.markdown('<div class="section-header">👤 步骤2: 选择待测试的人物</div>', unsafe_allow_html=True)
        
        tracks = st.session_state.tracks
        video_path = st.session_state.video_path
        
        # 显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("检测到的人物数", len(tracks))
        with col2:
            avg_frames = sum(len(t.frame_history) for t in tracks.values()) / len(tracks) if tracks else 0
            st.metric("平均跟踪帧数", f"{avg_frames:.1f}")
        with col3:
            best_track = max(tracks.values(), key=lambda t: len(t.frame_history))
            st.metric("最长轨迹帧数", len(best_track.frame_history))
        
        st.divider()
        
        # 显示每个轨迹的卡片
        st.subheader("检测到的人物轨迹")
        
        tracker = VideoPersonTracker()
        tracker.tracks = tracks
        tracker.frame_width = list(tracks.values())[0].frame_width if tracks else 1920
        tracker.frame_height = list(tracks.values())[0].frame_height if tracks else 1080
        
        # 创建网格布局
        cols = st.columns(3)
        
        for idx, (track_id, track) in enumerate(tracks.items()):
            with cols[idx % 3]:
                # 获取预览图
                preview_frame = tracker.get_track_preview_image(video_path, track)
                if preview_frame is not None:
                    preview_pil = numpy_to_pil(preview_frame)
                    
                    # 位置分布
                    pos_dist = get_position_distribution(track.position_history)
                    pos_text = f"左:{pos_dist['left']} 中:{pos_dist['center']} 右:{pos_dist['right']}"
                    
                    # 卡片样式
                    is_selected = st.session_state.selected_track_id == track_id
                    border_color = "#27ae60" if is_selected else "#3498db"
                    bg_color = "#e8f5e9" if is_selected else "#f8f9fa"
                    
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; border-radius: 10px; padding: 15px; 
                                margin: 10px 0; border-left: 4px solid {border_color};
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="margin: 0 0 10px 0;">人物 #{track_id} {'✓ 已选择' if is_selected else ''}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(preview_pil, use_container_width=True)
                    
                    st.markdown(f"""
                    <div style="font-size: 0.9rem; color: #666;">
                        <b>跟踪帧数:</b> {len(track.frame_history)}<br>
                        <b>位置分布:</b> {pos_text}<br>
                        <b>最佳质量:</b> {track.best_quality:.3f}<br>
                        <b>最佳帧:</b> #{track.best_frame}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"选择人物 #{track_id}", key=f"select_{track_id}", use_container_width=True):
                        st.session_state.selected_track_id = track_id
                        st.session_state.test_results = None
                        st.rerun()
        
        # 步骤3: 执行测试
        if st.session_state.selected_track_id is not None:
            selected_id = st.session_state.selected_track_id
            selected_track = tracks[selected_id]
            
            st.markdown('<div class="section-header">🧪 步骤3: CrossMedia-PID匹配测试</div>', unsafe_allow_html=True)
            
            st.info(f"已选择人物 #{selected_id}，跟踪帧数: {len(selected_track.frame_history)}")
            
            # 查找左右位置的帧
            pos_dist = get_position_distribution(selected_track.position_history)
            available_positions = [p for p, c in pos_dist.items() if c > 0]
            
            st.write(f"**可用位置:** {', '.join(available_positions)}")
            
            if len(available_positions) >= 2:
                if st.button("🔬 执行匹配测试", type="primary", use_container_width=True):
                    with st.spinner("正在初始化CrossMedia-PID系统..."):
                        if st.session_state.pid_system is None:
                            st.session_state.pid_system = init_crossmedia_pid()
                    
                    if st.session_state.pid_system is None:
                        st.error("CrossMedia-PID初始化失败")
                    else:
                        with st.spinner("正在提取位置截图..."):
                            # 提取左右位置的截图
                            positions_to_find = ["left", "right"]
                            if "left" not in available_positions or "right" not in available_positions:
                                # 如果没有左右都有，使用第一帧和最后一帧
                                positions_to_find = ["first", "last"]
                                frames_data = {}
                                
                                # 第一帧
                                frame_num = selected_track.frame_history[0]
                                bbox = selected_track.bbox_history[0]
                                crop = tracker.extract_frame_crop(video_path, frame_num, bbox)
                                frames_data["first"] = (frame_num, crop)
                                
                                # 最后一帧
                                frame_num = selected_track.frame_history[-1]
                                bbox = selected_track.bbox_history[-1]
                                crop = tracker.extract_frame_crop(video_path, frame_num, bbox)
                                frames_data["last"] = (frame_num, crop)
                            else:
                                frames_data = tracker.find_frames_at_positions(
                                    video_path, selected_id, ["left", "right"]
                                )
                        
                        if len(frames_data) >= 2:
                            with st.spinner("正在使用CrossMedia-PID处理..."):
                                results = []
                                for pos, (frame_num, crop) in frames_data.items():
                                    result = process_image_with_pid(
                                        st.session_state.pid_system,
                                        crop,
                                        f"{pos}_frame{frame_num}"
                                    )
                                    if result:
                                        results.append((pos, frame_num, crop, result))
                                
                                # 第一个添加到数据库
                                if len(results) >= 1:
                                    _, _, _, first_result = results[0]
                                    st.session_state.pid_system['matcher'].add_identity(
                                        person_uuid=first_result['person_uuid'],
                                        dense_vector=first_result['dense_vector'],
                                        sparse_vector=first_result['sparse_vector'],
                                        attributes=first_result['attributes'],
                                        source_meta={
                                            'source_path': first_result['source_name'],
                                            'quality_score': first_result['quality_score']
                                        }
                                    )
                                
                                st.session_state.test_results = results
                                st.rerun()
                        else:
                            st.error("无法提取足够的截图进行测试")
            else:
                st.warning("该人物的轨迹位置变化不足，请选择其他人物")
        
        # 显示测试结果
        if st.session_state.test_results:
            st.markdown('<div class="section-header">📊 测试结果</div>', unsafe_allow_html=True)
            
            results = st.session_state.test_results
            
            # 显示截图对比
            st.subheader("提取的测试截图")
            cols = st.columns(len(results))
            
            for idx, (pos, frame_num, crop, result) in enumerate(results):
                with cols[idx]:
                    st.image(numpy_to_pil(crop), caption=f"位置: {pos} (帧 #{frame_num})", use_container_width=True)
            
            # 显示匹配结果
            st.subheader("CrossMedia-PID识别结果")
            
            if len(results) >= 2:
                result1 = results[0][3]
                result2 = results[1][3]
                
                same_person = result1['person_uuid'] == result2['person_uuid']
                
                # 结果表格
                result_data = {
                    "属性": ["位置", "Person UUID", "是否新身份", "匹配分数", "质量分数", "处理时间"],
                    "图片1": [
                        results[0][0],
                        result1['person_uuid'][:16] + "...",
                        "是" if result1['is_new'] else "否",
                        "N/A (参考)",
                        f"{result1['quality_score']:.3f}",
                        f"{result1['elapsed_time']:.2f}s"
                    ],
                    "图片2": [
                        results[1][0],
                        result2['person_uuid'][:16] + "...",
                        "是" if result2['is_new'] else "否",
                        f"{result2['match_score']:.3f}",
                        f"{result2['quality_score']:.3f}",
                        f"{result2['elapsed_time']:.2f}s"
                    ]
                }
                
                st.table(result_data)
                
                # 测试结论
                st.divider()
                if same_person:
                    st.markdown('<div class="status-success">✅ 测试通过：成功识别为同一人！</div>', unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown('<div class="status-fail">❌ 测试失败：识别为不同人物</div>', unsafe_allow_html=True)
                
                # 显示属性对比
                with st.expander("查看详细属性对比"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**{results[0][0]} 属性:**")
                        st.json(result1['attributes'])
                    with col2:
                        st.write(f"**{results[1][0]} 属性:**")
                        st.json(result2['attributes'])


if __name__ == '__main__':
    main()
