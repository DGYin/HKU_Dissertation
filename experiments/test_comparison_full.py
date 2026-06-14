"""
完整功能测试：判断test_photo中的人物是否是同一个人
包含：VLM特征提取、向量化、匹配全流程
"""
import numpy as np
import cv2

from crossmedia_pid.config import load_config, project_path
from crossmedia_pid.core.feature_vlm import create_feature_extractor
from crossmedia_pid.core.vectorizer import DynamicVectorizer
from crossmedia_pid.core.matcher import IdentityMatcher
from crossmedia_pid.db.chroma_store import ChromaStore
from crossmedia_pid.core.extractor import PersonExtractor

# 优化后的VLM Prompt - 更详细描述衣服图案
ENHANCED_PERSON_FEATURE_PROMPT = """你是一个专业的人物特征分析专家，专门用于监控视频分析。请仔细观察输入图片中的人物，提取详细的结构化特征并严格返回JSON。

【必须提取的特征类目】：
- gender: 性别（男/女/未知）
- age_group: 年龄段（儿童/青年/中年/老年）
- height_build: 身高体型（高瘦/中等/矮胖等）

【上衣详细特征 - 重点关注】：
- topwear_color: 上衣主颜色
- topwear_type: 上衣类型（T恤/衬衫/外套/卫衣/夹克等）
- topwear_pattern_type: 图案类型（纯色/条纹/格子/印花/图形/文字/logo等）
- topwear_pattern_desc: 图案详细描述（如：胸前有白色大logo、袖子有条纹、背后有图案等）
- topwear_pattern_position: 图案位置（胸前/背后/左胸/右胸/袖子/下摆等）
- topwear_pattern_size: 图案大小（大/中/小/占据整个衣服等）
- topwear_pattern_color: 图案颜色

【下装详细特征】：
- bottomwear_color: 下装颜色
- bottomwear_type: 下装类型（长裤/短裤/裙子等）
- bottomwear_pattern: 下装图案描述（纯色/条纹/有装饰等）

【鞋子特征】：
- shoes_color: 鞋子颜色
- shoes_type: 鞋子类型（运动鞋/皮鞋/凉鞋/靴子等）

【配饰与携带物】：
- bag: 是否携带包（无/背包/手提包/挎包等，描述颜色和类型）
- glasses: 是否戴眼镜（无/眼镜类型和颜色）
- hat: 是否戴帽子（无/帽子类型和颜色）
- mask: 是否戴口罩

【头部特征】：
- hair_style: 发型（短发/长发/卷发/直发/光头等）
- hair_color: 发色（黑色/棕色/金色/灰色等）

【显著特征】：
- distinctive_traits: 显著特征（纹身位置、疤痕、特殊标记等）
- holding_object: 手持物品
- posture: 姿态（站立/行走/坐着/蹲着等）


【规则】
1. 对于衣服图案，必须详细描述：图案形状、位置、大小、颜色
2. 如果看到logo、文字、标志，必须描述其内容和位置
3. 若某项特征无法确定，值请严格填为 "无"
4. 仅输出合法JSON，不要包含任何解释性文字
5. 值使用中文描述，键名使用英文snake_case

请分析图片并输出JSON："""


def compare_persons_full():
    """完整功能比较test_photo中的人物"""
    
    # 加载配置
    config = load_config()
    
    print("🔍 开始完整功能测试 - 分析test_photo中的人物...")
    print("=" * 60)
    
    # 测试图片路径
    test_dir = project_path("experiments", "fixtures", "test_photo")
    image1_path = test_dir / "test_person_1.png"
    image2_path = test_dir / "test_person_2.png"
    
    if not image1_path.exists() or not image2_path.exists():
        print("❌ 图片不存在")
        return
    
    print(f"📷 图片1: {image1_path.name}")
    print(f"📷 图片2: {image2_path.name}")
    print()
    
    # ========== 初始化所有模块 ==========
    print("🚀 初始化系统模块...")
    
    # 1. 特征提取器（使用优化后的prompt）
    vlm_config = config.get('models', {}).get('vlm', {})
    vlm_config['prompt'] = ENHANCED_PERSON_FEATURE_PROMPT
    feature_extractor = create_feature_extractor(vlm_config)
    print("  ✅ VLM特征提取器初始化完成")
    
    # 2. 向量化器
    embedding_config = config.get('models', {}).get('embedding', {})
    registry_config = config.get('registry', {})
    vectorizer = DynamicVectorizer(
        dense_model_name=embedding_config.get('model_name', 'BAAI/bge-small-zh-v1.5'),
        max_length=embedding_config.get('max_length', 512),
        registry_path=registry_config.get('persist_path', str(project_path("data", "attribute_registry.json")))
    )
    print("  ✅ 向量化器初始化完成")
    
    # 3. 向量数据库
    chroma_config = config.get('database', {}).get('chroma', {})
    store = ChromaStore(
        persist_directory=chroma_config.get('persist_directory', str(project_path("data", "chroma_db"))),
        collection_name="test_comparison",
        distance_fn=chroma_config.get('distance_fn', 'cosine')
    )
    print("  ✅ ChromaDB初始化完成（使用独立测试集合）")
    
    # 4. 匹配器
    matching_config = config.get('matching', {})
    matcher = IdentityMatcher(
        store=store,
        threshold=matching_config.get('threshold', 0.72),
        top_k=matching_config.get('top_k', 5),
        weights=matching_config.get('weights'),
        enable_face=False
    )
    print("  ✅ 身份匹配器初始化完成")
    print()
    
    # ========== 处理第一张图片 ==========
    print("=" * 60)
    print("📸 处理第一张图片...")
    print("-" * 60)
    
    # 读取图片
    img1_cv = cv2.imread(str(image1_path))
    if img1_cv is None:
        print("❌ 图片1读取失败")
        return
    
    # 尝试检测人物（使用低阈值）
    extractor = PersonExtractor(
        conf_threshold=0.25,
        min_bbox_size=32
    )
    result1 = extractor.extract(image1_path, return_best_only=True)
    
    if result1 is None:
        print("  ⚠️  未检测到人物，使用整张图片进行VLM分析")
        img1_for_vlm = img1_cv
    else:
        print(f"  ✅ 检测到人物 (quality={result1.quality_score:.2f})")
        img1_for_vlm = result1.crop_image
    
    # VLM特征提取
    print("  🧠 使用VLM提取特征...")
    feature_output1 = feature_extractor.extract(img1_for_vlm)
    
    if not feature_output1.is_valid:
        print(f"  ❌ VLM特征提取失败: {feature_output1.raw_response[:100]}")
        return
    
    attributes1 = feature_output1.attributes
    print(f"  ✅ VLM提取成功，获得 {len(attributes1)} 个属性")
    
    # 显示提取的特征
    print("\n  📋 提取的特征详情:")
    for key, value in sorted(attributes1.items()):
        print(f"     {key}: {value}")
    
    # 向量化
    print("\n  🔢 执行向量化...")
    vector_output1 = vectorizer.vectorize(
        attributes1,
        source_meta={'source_path': str(image1_path), 'image': 'person_1'}
    )
    print(f"  ✅ 稠密向量维度: {len(vector_output1.dense_vector)}")
    print(f"  ✅ 稀疏向量维度: {len(vector_output1.sparse_vector)}")
    print(f"  ✅ 原始文本: {vector_output1.raw_text[:80]}...")
    
    # 添加到数据库（作为第一个身份）
    print("\n  💾 添加到数据库...")
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
    print(f"  ✅ 已添加身份: {match_result1.person_uuid}")
    print()
    
    # ========== 处理第二张图片 ==========
    print("=" * 60)
    print("📸 处理第二张图片...")
    print("-" * 60)
    
    # 读取图片
    img2_cv = cv2.imread(str(image2_path))
    if img2_cv is None:
        print("❌ 图片2读取失败")
        return
    
    # 尝试检测人物
    result2 = extractor.extract(image2_path, return_best_only=True)
    
    if result2 is None:
        print("  ⚠️  未检测到人物，使用整张图片进行VLM分析")
        img2_for_vlm = img2_cv
    else:
        print(f"  ✅ 检测到人物 (quality={result2.quality_score:.2f})")
        img2_for_vlm = result2.crop_image
    
    # VLM特征提取
    print("  🧠 使用VLM提取特征...")
    feature_output2 = feature_extractor.extract(img2_for_vlm)
    
    if not feature_output2.is_valid:
        print(f"  ❌ VLM特征提取失败: {feature_output2.raw_response[:100]}")
        return
    
    attributes2 = feature_output2.attributes
    print(f"  ✅ VLM提取成功，获得 {len(attributes2)} 个属性")
    
    # 显示提取的特征
    print("\n  📋 提取的特征详情:")
    for key, value in sorted(attributes2.items()):
        print(f"     {key}: {value}")
    
    # 向量化
    print("\n  🔢 执行向量化...")
    vector_output2 = vectorizer.vectorize(
        attributes2,
        source_meta={'source_path': str(image2_path), 'image': 'person_2'}
    )
    print(f"  ✅ 稠密向量维度: {len(vector_output2.dense_vector)}")
    print(f"  ✅ 稀疏向量维度: {len(vector_output2.sparse_vector)}")
    
    # 执行匹配（与数据库中的身份比较）
    print("\n  🔍 执行身份匹配...")
    match_result2 = matcher.match(
        dense_vector=vector_output2.dense_vector,
        sparse_vector=vector_output2.sparse_vector,
        query_attributes=attributes2
    )
    
    print(f"  ✅ 匹配完成")
    print(f"     - 匹配到的身份: {match_result2.person_uuid}")
    print(f"     - 是否新身份: {'是' if match_result2.is_new_identity else '否'}")
    print(f"     - 综合匹配分数: {match_result2.match_score:.4f}")
    print(f"     - 稠密向量相似度: {match_result2.dense_score:.4f}")
    print(f"     - 稀疏向量相似度: {match_result2.sparse_score:.4f}")
    print()
    
    # ========== 详细比较分析 ==========
    print("=" * 60)
    print("🔬 详细特征比较分析")
    print("=" * 60)
    
    # 1. 衣服图案详细比较
    print("\n👕 上衣图案详细对比:")
    clothing_attrs = [
        'topwear_color', 'topwear_type', 'topwear_pattern_type',
        'topwear_pattern_desc', 'topwear_pattern_position',
        'topwear_pattern_size', 'topwear_pattern_color'
    ]
    
    for attr in clothing_attrs:
        val1 = attributes1.get(attr, '无')
        val2 = attributes2.get(attr, '无')
        match = str(val1).lower() == str(val2).lower()
        symbol = "✅" if match else "❌"
        print(f"  {symbol} {attr}:")
        print(f"     图1: {val1}")
        print(f"     图2: {val2}")
    
    # 2. 所有共同属性比较
    common_attrs = set(attributes1.keys()) & set(attributes2.keys())
    print(f"\n📊 所有共同属性比较 ({len(common_attrs)} 个):")
    
    matching_count = 0
    key_matching_count = 0
    key_total = 0
    
    # 定义关键属性
    key_attributes = [
        'gender', 'age_group', 'topwear_color', 'topwear_pattern_type',
        'topwear_pattern_desc', 'bottomwear_color', 'hair_style'
    ]
    
    for attr in sorted(common_attrs):
        val1 = attributes1.get(attr, 'N/A')
        val2 = attributes2.get(attr, 'N/A')
        match = str(val1).lower() == str(val2).lower()
        
        if match:
            matching_count += 1
        
        if attr in key_attributes:
            key_total += 1
            if match:
                key_matching_count += 1
            is_key = " [关键]"
        else:
            is_key = ""
        
        symbol = "✅" if match else "❌"
        print(f"  {symbol} {attr}{is_key}: '{val1}' vs '{val2}'")
    
    # 3. 计算各种匹配度
    print("\n📈 匹配度统计:")
    if common_attrs:
        overall_match_ratio = matching_count / len(common_attrs)
        print(f"  总体属性匹配度: {matching_count}/{len(common_attrs)} ({overall_match_ratio:.1%})")
    
    if key_total > 0:
        key_match_ratio = key_matching_count / key_total
        print(f"  关键属性匹配度: {key_matching_count}/{key_total} ({key_match_ratio:.1%})")
    
    # 4. 向量相似度计算
    print("\n🔢 向量相似度分析:")
    
    # 稠密向量余弦相似度
    dense_vec1 = np.array(vector_output1.dense_vector)
    dense_vec2 = np.array(vector_output2.dense_vector)
    dense_sim = np.dot(dense_vec1, dense_vec2) / (np.linalg.norm(dense_vec1) * np.linalg.norm(dense_vec2))
    print(f"  稠密向量余弦相似度: {dense_sim:.4f}")
    
    # 稀疏向量Jaccard相似度
    sparse1 = vector_output1.sparse_vector
    sparse2 = vector_output2.sparse_vector
    keys1 = set(sparse1.keys())
    keys2 = set(sparse2.keys())
    intersection = keys1 & keys2
    union = keys1 | keys2
    jaccard_sim = len(intersection) / len(union) if union else 0
    print(f"  稀疏向量Jaccard相似度: {jaccard_sim:.4f}")
    print(f"    - 共有属性维度: {len(intersection)}")
    print(f"    - 总属性维度: {len(union)}")
    
    # 5. 系统匹配结果
    print("\n🎯 系统匹配决策:")
    print(f"  匹配阈值: {matcher.threshold}")
    print(f"  综合匹配分数: {match_result2.match_score:.4f}")
    print(f"  权重配置: 稠密={matcher.weights['dense']:.2f}, 稀疏={matcher.weights['sparse']:.2f}")
    
    if match_result2.is_new_identity:
        print(f"\n  ❌ 系统判断: 这是不同的人（创建了新身份）")
        print(f"     原因: 匹配分数 {match_result2.match_score:.4f} < 阈值 {matcher.threshold}")
    else:
        print(f"\n  ✅ 系统判断: 这是同一个人")
        print(f"     匹配到的身份ID: {match_result2.person_uuid}")
    
    # 6. 候选列表
    if match_result2.top_candidates:
        print(f"\n📋 Top候选匹配:")
        for i, cand in enumerate(match_result2.top_candidates[:3], 1):
            print(f"  {i}. {cand['person_uuid']}: {cand['total_score']:.4f} "
                  f"(dense={cand['dense_similarity']:.4f}, sparse={cand['sparse_similarity']:.4f})")
    
    # ========== 最终结论 ==========
    print("\n" + "=" * 60)
    print("🏁 最终结论")
    print("=" * 60)
    
    # 综合判断（结合系统结果和属性分析）
    if match_result2.is_new_identity:
        conclusion = "这两张图片中的人物不是同一个人"
        confidence = "高"
    else:
        conclusion = "这两张图片中的人物是同一个人"
        confidence = "中"
    
    print(f"\n系统结论: {conclusion}")
    print(f"置信度: {confidence}")
    print(f"\n详细依据:")
    print(f"  - VLM提取的属性数量: 图1={len(attributes1)}, 图2={len(attributes2)}")
    print(f"  - 共同属性数量: {len(common_attrs)}")
    print(f"  - 属性匹配率: {overall_match_ratio:.1%}" if common_attrs else "  - 属性匹配率: N/A")
    print(f"  - 稠密向量相似度: {dense_sim:.4f}")
    print(f"  - 稀疏向量相似度: {jaccard_sim:.4f}")
    print(f"  - 系统综合匹配分数: {match_result2.match_score:.4f}")
    
    # 衣服图案差异分析
    pattern1 = attributes1.get('topwear_pattern_desc', '无')
    pattern2 = attributes2.get('topwear_pattern_desc', '无')
    if pattern1 != pattern2 and pattern1 != '无' and pattern2 != '无':
        print(f"\n⚠️  注意: 衣服图案描述存在显著差异")
        print(f"  图1: {pattern1}")
        print(f"  图2: {pattern2}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    compare_persons_full()
