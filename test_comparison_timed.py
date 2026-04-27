"""
带计时功能的完整测试：判断test_photo中的人物是否是同一个人
包含：性能监控、耗时统计、自动报警功能
"""
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent / "crossmedia_pid"
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import cv2
from main import load_config
from core.feature_vlm import create_feature_extractor, FeatureOutput
from core.vectorizer import DynamicVectorizer
from core.matcher import IdentityMatcher
from db.chroma_store import ChromaStore
from core.extractor import PersonExtractor


@dataclass
class TimingRecord:
    """单次计时记录"""
    step_name: str
    start_time: float
    end_time: float
    details: Dict = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class PerformanceMonitor:
    """性能监控器 - 记录和分析各步骤耗时"""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化性能监控器
        
        Args:
            alert_thresholds: 各步骤的报警阈值（秒）
        """
        self.records: List[TimingRecord] = []
        self.current_timers: Dict[str, float] = {}
        
        # 默认报警阈值（秒）
        self.alert_thresholds = alert_thresholds or {
            'image_loading': 1.0,
            'person_detection': 5.0,
            'vlm_feature_extraction': 15.0,
            'vectorization': 10.0,
            'database_search': 3.0,
            'identity_matching': 2.0,
            'total_processing': 30.0
        }
        
        # 统计信息
        self.step_stats: Dict[str, List[float]] = defaultdict(list)
    
    def start_timer(self, step_name: str):
        """开始计时"""
        self.current_timers[step_name] = time.time()
        print(f"  ⏱️  [{step_name}] 开始...")
    
    def end_timer(self, step_name: str, details: Optional[Dict] = None) -> float:
        """结束计时并记录"""
        if step_name not in self.current_timers:
            print(f"  ⚠️  计时器 {step_name} 未启动")
            return 0.0
        
        end_time = time.time()
        start_time = self.current_timers.pop(step_name)
        duration = end_time - start_time
        
        # 创建记录
        record = TimingRecord(
            step_name=step_name,
            start_time=start_time,
            end_time=end_time,
            details=details or {}
        )
        self.records.append(record)
        
        # 更新统计
        self.step_stats[step_name].append(duration)
        
        # 显示耗时
        status = "✅" if duration < self.alert_thresholds.get(step_name, float('inf')) else "⚠️"
        threshold_info = ""
        if step_name in self.alert_thresholds:
            threshold_info = f" (阈值: {self.alert_thresholds[step_name]:.2f}s)"
        
        print(f"  {status} [{step_name}] 完成，耗时: {duration:.3f}s{threshold_info}")
        
        # 报警检查
        if step_name in self.alert_thresholds and duration > self.alert_thresholds[step_name]:
            self._alert(step_name, duration)
        
        return duration
    
    def _alert(self, step_name: str, duration: float):
        """触发报警"""
        threshold = self.alert_thresholds[step_name]
        print(f"  🚨 [报警] {step_name} 耗时过长: {duration:.3f}s > {threshold:.3f}s")
    
    def get_step_statistics(self, step_name: str) -> Dict:
        """获取某步骤的统计信息"""
        times = self.step_stats.get(step_name, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times)
        }
    
    def print_summary(self):
        """打印性能总结"""
        print("\n" + "=" * 70)
        print("📊 性能监控总结")
        print("=" * 70)
        
        # 按步骤分组统计
        all_steps = set(record.step_name for record in self.records)
        
        print("\n⏱️  各步骤耗时统计:")
        print("-" * 70)
        print(f"{'步骤名称':<30} {'次数':>6} {'平均(s)':>10} {'最小(s)':>10} {'最大(s)':>10} {'状态':>8}")
        print("-" * 70)
        
        for step_name in sorted(all_steps):
            stats = self.get_step_statistics(step_name)
            if stats:
                threshold = self.alert_thresholds.get(step_name, float('inf'))
                status = "✅ 正常" if stats['mean'] < threshold else "⚠️ 超时"
                print(f"{step_name:<30} {stats['count']:>6} {stats['mean']:>10.3f} "
                      f"{stats['min']:>10.3f} {stats['max']:>10.3f} {status:>8}")
        
        # 总耗时
        total_time = sum(r.duration for r in self.records)
        print("-" * 70)
        print(f"{'总耗时':<30} {'':>6} {total_time:>10.3f}")
        print()
        
        # 报警汇总
        alerts = [r for r in self.records 
                 if r.step_name in self.alert_thresholds 
                 and r.duration > self.alert_thresholds[r.step_name]]
        
        if alerts:
            print(f"🚨 报警汇总 ({len(alerts)} 次超时):")
            for record in alerts:
                threshold = self.alert_thresholds[record.step_name]
                print(f"   - {record.step_name}: {record.duration:.3f}s > {threshold:.3f}s")
        else:
            print("✅ 所有步骤均在正常时间范围内完成")
        
        print()
    
    def export_stats_for_visualization(self) -> Dict:
        """
        导出统计数据供可视化使用
        
        Returns:
            包含所有统计数据的字典
        """
        data = {
            'timestamp': time.time(),
            'total_records': len(self.records),
            'steps': {}
        }
        
        for step_name in self.step_stats.keys():
            stats = self.get_step_statistics(step_name)
            data['steps'][step_name] = {
                'count': stats['count'],
                'mean_duration': round(stats['mean'], 3),
                'min_duration': round(stats['min'], 3),
                'max_duration': round(stats['max'], 3),
                'std_duration': round(stats['std'], 3),
                'alert_threshold': self.alert_thresholds.get(step_name),
                'alert_count': sum(1 for r in self.records 
                                  if r.step_name == step_name 
                                  and r.step_name in self.alert_thresholds
                                  and r.duration > self.alert_thresholds[r.step_name])
            }
        
        return data


def compare_persons_with_timing():
    """带计时功能的完整比较"""
    
    # 初始化性能监控器
    monitor = PerformanceMonitor(alert_thresholds={
        'image_loading': 0.5,
        'person_detection': 3.0,
        'vlm_feature_extraction': 10.0,
        'vectorization': 5.0,
        'database_search': 2.0,
        'identity_matching': 1.0,
        'total_processing': 25.0
    })
    
    print("🔍 开始带计时功能的完整测试...")
    print("=" * 70)
    
    # 加载配置
    config_path = project_root / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    # 测试图片路径
    test_dir = Path(__file__).parent / "test_photo"
    image1_path = test_dir / "test_person_1.png"
    image2_path = test_dir / "test_person_2.png"
    
    print(f"📷 图片1: {image1_path.name}")
    print(f"📷 图片2: {image2_path.name}")
    print()
    
    # ========== 初始化模块（带计时） ==========
    print("🚀 初始化系统模块...")
    monitor.start_timer('module_initialization')
    
    # 1. 特征提取器
    vlm_config = config.get('models', {}).get('vlm', {})
    feature_extractor = create_feature_extractor(vlm_config)
    
    # 2. 向量化器
    embedding_config = config.get('models', {}).get('embedding', {})
    registry_config = config.get('registry', {})
    vectorizer = DynamicVectorizer(
        dense_model_name=embedding_config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
        max_length=embedding_config.get('max_length', 512),
        registry_path=registry_config.get('persist_path', './attribute_registry.json')
    )
    
    # 3. 向量数据库
    chroma_config = config.get('database', {}).get('chroma', {})
    store = ChromaStore(
        persist_directory=chroma_config.get('persist_directory', './chroma_db'),
        collection_name="test_comparison_timed",
        distance_fn=chroma_config.get('distance_fn', 'cosine')
    )
    
    # 4. 匹配器
    matching_config = config.get('matching', {})
    matcher = IdentityMatcher(
        store=store,
        threshold=matching_config.get('threshold', 0.72),
        top_k=matching_config.get('top_k', 5),
        weights=matching_config.get('weights'),
        enable_face=False
    )
    
    monitor.end_timer('module_initialization')
    print()
    
    # ========== 处理第一张图片 ==========
    print("=" * 70)
    print("📸 处理第一张图片")
    print("=" * 70)
    
    # 1. 加载图片
    monitor.start_timer('image_loading')
    img1_cv = cv2.imread(str(image1_path))
    monitor.end_timer('image_loading', {'image_path': str(image1_path), 'success': img1_cv is not None})
    
    # 2. 人物检测
    monitor.start_timer('person_detection')
    extractor = PersonExtractor(conf_threshold=0.25, min_bbox_size=32)
    result1 = extractor.extract(image1_path, return_best_only=True)
    monitor.end_timer('person_detection', {
        'detected': result1 is not None,
        'quality_score': result1.quality_score if result1 else None
    })
    
    img1_for_vlm = result1.crop_image if result1 else img1_cv
    
    # 3. VLM特征提取
    monitor.start_timer('vlm_feature_extraction')
    feature_output1 = feature_extractor.extract(img1_for_vlm)
    monitor.end_timer('vlm_feature_extraction', {
        'success': feature_output1.is_valid,
        'attribute_count': len(feature_output1.attributes) if feature_output1.is_valid else 0
    })
    
    if not feature_output1.is_valid:
        print("❌ 第一张图片特征提取失败")
        return
    
    attributes1 = feature_output1.attributes
    print(f"  📋 提取到 {len(attributes1)} 个属性")
    
    # 4. 向量化
    monitor.start_timer('vectorization')
    vector_output1 = vectorizer.vectorize(
        attributes1,
        source_meta={'source_path': str(image1_path), 'image': 'person_1'}
    )
    monitor.end_timer('vectorization', {
        'dense_dim': len(vector_output1.dense_vector),
        'sparse_dim': len(vector_output1.sparse_vector)
    })
    
    # 5. 添加到数据库
    monitor.start_timer('database_add')
    match_result1 = matcher.match(
        dense_vector=vector_output1.dense_vector,
        sparse_vector=vector_output1.sparse_vector,
        query_attributes=attributes1
    )
    doc_id1 = matcher.add_identity(
        person_uuid=match_result1.person_uuid,
        dense_vector=vector_output1.dense_vector,
        sparse_vector=vector_output1.sparse_vector,
        attributes=attributes1,
        source_meta={'source_path': str(image1_path)}
    )
    monitor.end_timer('database_add', {'person_uuid': match_result1.person_uuid})
    print(f"  ✅ 身份 {match_result1.person_uuid} 已添加到数据库")
    print()
    
    # ========== 处理第二张图片 ==========
    print("=" * 70)
    print("📸 处理第二张图片")
    print("=" * 70)
    
    # 1. 加载图片
    monitor.start_timer('image_loading')
    img2_cv = cv2.imread(str(image2_path))
    monitor.end_timer('image_loading', {'image_path': str(image2_path), 'success': img2_cv is not None})
    
    # 2. 人物检测
    monitor.start_timer('person_detection')
    result2 = extractor.extract(image2_path, return_best_only=True)
    monitor.end_timer('person_detection', {
        'detected': result2 is not None,
        'quality_score': result2.quality_score if result2 else None
    })
    
    img2_for_vlm = result2.crop_image if result2 else img2_cv
    
    # 3. VLM特征提取
    monitor.start_timer('vlm_feature_extraction')
    feature_output2 = feature_extractor.extract(img2_for_vlm)
    monitor.end_timer('vlm_feature_extraction', {
        'success': feature_output2.is_valid,
        'attribute_count': len(feature_output2.attributes) if feature_output2.is_valid else 0
    })
    
    if not feature_output2.is_valid:
        print("❌ 第二张图片特征提取失败")
        return
    
    attributes2 = feature_output2.attributes
    print(f"  📋 提取到 {len(attributes2)} 个属性")
    
    # 4. 向量化
    monitor.start_timer('vectorization')
    vector_output2 = vectorizer.vectorize(
        attributes2,
        source_meta={'source_path': str(image2_path), 'image': 'person_2'}
    )
    monitor.end_timer('vectorization', {
        'dense_dim': len(vector_output2.dense_vector),
        'sparse_dim': len(vector_output2.sparse_vector)
    })
    
    # 5. 身份匹配
    monitor.start_timer('identity_matching')
    match_result2 = matcher.match(
        dense_vector=vector_output2.dense_vector,
        sparse_vector=vector_output2.sparse_vector,
        query_attributes=attributes2
    )
    monitor.end_timer('identity_matching', {
        'is_new_identity': match_result2.is_new_identity,
        'match_score': match_result2.match_score,
        'matched_uuid': match_result2.person_uuid
    })
    
    print(f"  ✅ 匹配完成")
    print(f"     - 匹配到的身份: {match_result2.person_uuid}")
    print(f"     - 是否新身份: {'是' if match_result2.is_new_identity else '否'}")
    print(f"     - 综合匹配分数: {match_result2.match_score:.4f}")
    print()
    
    # ========== 打印性能总结 ==========
    monitor.print_summary()
    
    # ========== 导出可视化数据 ==========
    viz_data = monitor.export_stats_for_visualization()
    print("📤 可视化数据已准备（可通过API导出）:")
    print(f"   总记录数: {viz_data['total_records']}")
    print(f"   监控步骤: {list(viz_data['steps'].keys())}")
    
    # 保存到文件（可选）
    import json
    stats_file = Path(__file__).parent / "performance_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)
    print(f"   统计数据已保存到: {stats_file}")
    print()
    
    # ========== 最终结论 ==========
    print("=" * 70)
    print("🏁 最终结论")
    print("=" * 70)
    
    if match_result2.is_new_identity:
        print("❌ 这两张图片中的人物不是同一个人")
    else:
        print("✅ 这两张图片中的人物是同一个人")
    
    print(f"\n匹配分数: {match_result2.match_score:.4f}")
    print(f"稠密向量相似度: {match_result2.dense_score:.4f}")
    print(f"稀疏向量相似度: {match_result2.sparse_score:.4f}")
    print()


if __name__ == "__main__":
    compare_persons_with_timing()
