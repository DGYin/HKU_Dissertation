"""
稳定性测试：连续运行10次人物识别测试
统计系统判断的一致性和稳定性
"""
import sys
import time
from pathlib import Path
from collections import defaultdict
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent / "crossmedia_pid"
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from main import load_config
from core.feature_vlm import create_feature_extractor
from core.vectorizer import DynamicVectorizer
from core.matcher import IdentityMatcher
from db.chroma_store import ChromaStore
from core.extractor import PersonExtractor


def run_single_test(test_id: int, config: dict, image1_path: Path, image2_path: Path, 
                   feature_extractor, vectorizer, store, matcher):
    """
    运行单次测试
    
    Returns:
        dict: 测试结果
    """
    print(f"\n{'='*70}")
    print(f"🧪 测试 #{test_id + 1}/10")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # 清空数据库（每次测试独立）
    try:
        store._init_client()
        all_ids = store._collection.get()['ids']
        if all_ids:
            store._collection.delete(ids=all_ids)
    except:
        pass
    
    extractor = PersonExtractor(conf_threshold=0.25, min_bbox_size=32)
    
    # ========== 处理第一张图片 ==========
    img1 = cv2.imread(str(image1_path))
    result1 = extractor.extract(image1_path, return_best_only=True)
    img1_for_vlm = result1.crop_image if result1 else img1
    
    feature1 = feature_extractor.extract(img1_for_vlm)
    if not feature1.is_valid:
        return {'success': False, 'error': '图片1特征提取失败'}
    
    vector1 = vectorizer.vectorize(feature1.attributes)
    
    match1 = matcher.match(
        dense_vector=vector1.dense_vector,
        sparse_vector=vector1.sparse_vector,
        query_attributes=feature1.attributes
    )
    
    matcher.add_identity(
        person_uuid=match1.person_uuid,
        dense_vector=vector1.dense_vector,
        sparse_vector=vector1.sparse_vector,
        attributes=feature1.attributes,
        source_meta={'source_path': str(image1_path)}
    )
    
    uuid1 = match1.person_uuid
    
    # ========== 处理第二张图片 ==========
    img2 = cv2.imread(str(image2_path))
    result2 = extractor.extract(image2_path, return_best_only=True)
    img2_for_vlm = result2.crop_image if result2 else img2
    
    feature2 = feature_extractor.extract(img2_for_vlm)
    if not feature2.is_valid:
        return {'success': False, 'error': '图片2特征提取失败'}
    
    vector2 = vectorizer.vectorize(feature2.attributes)
    
    match2 = matcher.match(
        dense_vector=vector2.dense_vector,
        sparse_vector=vector2.sparse_vector,
        query_attributes=feature2.attributes
    )
    
    uuid2 = match2.person_uuid
    
    elapsed = time.time() - start_time
    
    # 判断是否同一人
    is_same_person = not match2.is_new_identity
    
    # 统计属性匹配
    attrs1 = feature1.attributes
    attrs2 = feature2.attributes
    common_attrs = set(attrs1.keys()) & set(attrs2.keys())
    matching_attrs = sum(1 for k in common_attrs if str(attrs1[k]).lower() == str(attrs2[k]).lower())
    
    # 上衣图案关键特征
    pattern_attrs = ['topwear_pattern_color', 'topwear_pattern_desc', 'topwear_pattern_position']
    pattern_matching = sum(1 for k in pattern_attrs if k in attrs1 and k in attrs2 
                          and str(attrs1[k]).lower() == str(attrs2[k]).lower())
    
    result = {
        'test_id': test_id + 1,
        'success': True,
        'elapsed_time': elapsed,
        'is_same_person': is_same_person,
        'uuid1': uuid1,
        'uuid2': uuid2,
        'match_score': match2.match_score,
        'dense_score': match2.dense_score,
        'sparse_score': match2.sparse_score,
        'attribute_count': len(attrs1),
        'common_attributes': len(common_attrs),
        'matching_attributes': matching_attrs,
        'attribute_match_rate': matching_attrs / len(common_attrs) if common_attrs else 0,
        'pattern_match_count': pattern_matching,
        'pattern_match_rate': pattern_matching / len(pattern_attrs),
        'topwear_pattern_color_1': attrs1.get('topwear_pattern_color', 'N/A'),
        'topwear_pattern_color_2': attrs2.get('topwear_pattern_color', 'N/A'),
        'topwear_pattern_desc_1': attrs1.get('topwear_pattern_desc', 'N/A'),
        'topwear_pattern_desc_2': attrs2.get('topwear_pattern_desc', 'N/A'),
    }
    
    # 打印本次结果
    print(f"⏱️  耗时: {elapsed:.2f}s")
    print(f"🎯 判断: {'同一人' if is_same_person else '不同人'}")
    print(f"📊 匹配分数: {result['match_score']:.4f}")
    print(f"📋 属性匹配率: {result['attribute_match_rate']:.1%}")
    print(f"👕 图案匹配: {pattern_matching}/{len(pattern_attrs)}")
    print(f"   图1图案颜色: {result['topwear_pattern_color_1']}")
    print(f"   图2图案颜色: {result['topwear_pattern_color_2']}")
    
    return result


def run_stability_test():
    """运行稳定性测试（10次）"""
    
    print("🔬 稳定性测试：连续运行10次人物识别")
    print("=" * 70)
    
    # 加载配置
    config_path = project_root / "configs" / "config.yaml"
    config = load_config(str(config_path))
    
    # 测试图片
    test_dir = Path(__file__).parent / "test_photo"
    image1_path = test_dir / "test_person_1.png"
    image2_path = test_dir / "test_person_2.png"
    
    print(f"📷 图片1: {image1_path.name}")
    print(f"📷 图片2: {image2_path.name}")
    print()
    
    # 初始化系统（只初始化一次）
    print("🚀 初始化系统...")
    vlm_config = config.get('models', {}).get('vlm', {})
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
        collection_name="stability_test",
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
    print("✅ 系统初始化完成\n")
    
    # 运行10次测试
    results = []
    for i in range(10):
        try:
            result = run_single_test(
                test_id=i,
                config=config,
                image1_path=image1_path,
                image2_path=image2_path,
                feature_extractor=feature_extractor,
                vectorizer=vectorizer,
                store=store,
                matcher=matcher
            )
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 #{i+1} 失败: {str(e)}")
            results.append({
                'test_id': i + 1,
                'success': False,
                'error': str(e)
            })
        
        # 测试间延迟，避免API限流
        if i < 9:
            time.sleep(1)
    
    # ========== 统计分析 ==========
    print("\n" + "=" * 70)
    print("📊 稳定性测试统计分析")
    print("=" * 70)
    
    successful_results = [r for r in results if r.get('success')]
    failed_count = len(results) - len(successful_results)
    
    print(f"\n✅ 成功次数: {len(successful_results)}/10")
    if failed_count > 0:
        print(f"❌ 失败次数: {failed_count}/10")
    
    if not successful_results:
        print("⚠️  没有成功的测试结果")
        return
    
    # 判断一致性
    same_person_count = sum(1 for r in successful_results if r['is_same_person'])
    different_person_count = len(successful_results) - same_person_count
    
    print(f"\n🎯 系统判断统计:")
    print(f"   判断为同一人: {same_person_count} 次 ({same_person_count/len(successful_results)*100:.1f}%)")
    print(f"   判断为不同人: {different_person_count} 次 ({different_person_count/len(successful_results)*100:.1f}%)")
    
    # 一致性评估
    consistency = max(same_person_count, different_person_count) / len(successful_results)
    print(f"\n📈 判断一致性: {consistency:.1%}")
    
    if consistency >= 0.9:
        consistency_level = "⭐⭐⭐ 极高"
    elif consistency >= 0.7:
        consistency_level = "⭐⭐ 较高"
    elif consistency >= 0.5:
        consistency_level = "⭐ 一般"
    else:
        consistency_level = "⚠️ 低（系统不稳定）"
    print(f"   一致性等级: {consistency_level}")
    
    # 匹配分数统计
    match_scores = [r['match_score'] for r in successful_results]
    print(f"\n📊 匹配分数统计:")
    print(f"   平均值: {np.mean(match_scores):.4f}")
    print(f"   标准差: {np.std(match_scores):.4f}")
    print(f"   最小值: {np.min(match_scores):.4f}")
    print(f"   最大值: {np.max(match_scores):.4f}")
    
    # 属性匹配率统计
    attr_match_rates = [r['attribute_match_rate'] for r in successful_results]
    print(f"\n📋 属性匹配率统计:")
    print(f"   平均值: {np.mean(attr_match_rates):.1%}")
    print(f"   标准差: {np.std(attr_match_rates):.2%}")
    
    # 图案匹配统计
    pattern_match_rates = [r['pattern_match_rate'] for r in successful_results]
    print(f"\n👕 上衣图案匹配统计:")
    print(f"   平均匹配: {np.mean(pattern_match_rates):.1%}")
    
    # 耗时统计
    elapsed_times = [r['elapsed_time'] for r in successful_results]
    print(f"\n⏱️  耗时统计:")
    print(f"   总耗时: {sum(elapsed_times):.2f}s")
    print(f"   平均单次: {np.mean(elapsed_times):.2f}s")
    print(f"   最小: {np.min(elapsed_times):.2f}s")
    print(f"   最大: {np.max(elapsed_times):.2f}s")
    
    # 图案颜色一致性
    pattern_colors_1 = [r['topwear_pattern_color_1'] for r in successful_results]
    pattern_colors_2 = [r['topwear_pattern_color_2'] for r in successful_results]
    
    unique_colors_1 = set(pattern_colors_1)
    unique_colors_2 = set(pattern_colors_2)
    
    print(f"\n🎨 VLM图案颜色识别稳定性:")
    print(f"   图片1识别结果: {len(unique_colors_1)} 种不同描述")
    print(f"      {unique_colors_1}")
    print(f"   图片2识别结果: {len(unique_colors_2)} 种不同描述")
    print(f"      {unique_colors_2}")
    
    # ========== 详细结果表格 ==========
    print("\n" + "=" * 70)
    print("📋 详细测试结果")
    print("=" * 70)
    print(f"{'#':>3} | {'判断':>6} | {'匹配分':>8} | {'稠密':>6} | {'稀疏':>6} | {'属性率':>7} | {'图案':>4} | {'耗时':>6}")
    print("-" * 70)
    
    for r in successful_results:
        judgment = "同一人" if r['is_same_person'] else "不同"
        print(f"{r['test_id']:>3} | {judgment:>6} | {r['match_score']:>8.4f} | "
              f"{r['dense_score']:>6.4f} | {r['sparse_score']:>6.4f} | "
              f"{r['attribute_match_rate']:>7.1%} | {r['pattern_match_count']:>4} | "
              f"{r['elapsed_time']:>6.2f}")
    
    # ========== 最终结论 ==========
    print("\n" + "=" * 70)
    print("🏁 最终结论")
    print("=" * 70)
    
    dominant_judgment = "同一人" if same_person_count > different_person_count else "不同人"
    print(f"\n🎯 系统主要判断: {dominant_judgment} ({max(same_person_count, different_person_count)}/{len(successful_results)} 次)")
    
    if consistency < 0.7:
        print("⚠️  警告: 系统判断一致性较低，建议检查:")
        print("   - VLM API的稳定性")
        print("   - 匹配阈值设置")
        print("   - 图片质量")
    else:
        print("✅ 系统判断相对稳定")
    
    # 保存详细结果
    output_file = Path(__file__).parent / "stability_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_count': 10,
            'successful_count': len(successful_results),
            'consistency': consistency,
            'dominant_judgment': dominant_judgment,
            'same_person_count': same_person_count,
            'different_person_count': different_person_count,
            'detailed_results': successful_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存: {output_file}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_stability_test()
