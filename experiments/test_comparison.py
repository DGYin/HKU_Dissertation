"""
测试脚本：判断test_photo中的人物是否是同一个人
"""
import cv2

from crossmedia_pid.config import load_config, project_path
from crossmedia_pid.core.feature_vlm import create_feature_extractor
from crossmedia_pid.core.extractor import PersonExtractor

def compare_persons():
    """比较test_photo中的人物"""
    
    # 加载配置
    config = load_config()
    
    print("🔍 开始分析test_photo中的人物...")
    print("=" * 50)
    
    # 测试图片路径
    test_dir = project_path("experiments", "fixtures", "test_photo")
    image1_path = test_dir / "test_person_1.png"
    image2_path = test_dir / "test_person_2.png"
    
    if not image1_path.exists():
        print(f"❌ 图片不存在: {image1_path}")
        return
    
    if not image2_path.exists():
        print(f"❌ 图片不存在: {image2_path}")
        return
    
    print(f"📷 图片1: {image1_path}")
    print(f"📷 图片2: {image2_path}")
    print()
    
    # 初始化特征提取器（VLM）
    vlm_config = config.get('models', {}).get('vlm', {})
    feature_extractor = create_feature_extractor(vlm_config)
    
    # 读取图片
    img1_cv = cv2.imread(str(image1_path))
    img2_cv = cv2.imread(str(image2_path))
    
    if img1_cv is None or img2_cv is None:
        print("❌ 图片读取失败")
        return
    
    # 尝试使用低阈值检测人物
    print("尝试检测人物...")
    extractor_low_conf = PersonExtractor(
        conf_threshold=0.25,  # 降低置信度阈值
        min_bbox_size=32  # 降低最小尺寸
    )
    
    # 处理第一张图片
    print("\n处理第一张图片...")
    result1 = extractor_low_conf.extract(image1_path, return_best_only=True)
    
    if result1 is None:
        print("  ⚠️  未检测到人物，使用整张图片进行分析")
        # 使用整张图片
        feature_output1 = feature_extractor.extract(img1_cv)
        attributes1 = feature_output1.attributes
        quality1 = 0.5  # 默认质量
        print(f"  ✅ VLM特征提取成功 ({len(attributes1)} 个属性)")
    else:
        print(f"  ✅ 检测到人物 (quality={result1.quality_score:.2f})")
        # 使用裁剪后的人物图片
        feature_output1 = feature_extractor.extract(result1.crop_image)
        attributes1 = feature_output1.attributes
        quality1 = result1.quality_score
        print(f"  ✅ VLM特征提取成功 ({len(attributes1)} 个属性)")
    
    if not feature_output1.is_valid:
        print("❌ 第一张图片特征提取失败")
        return
    
    # 显示第一张图片的属性
    print("\n📋 第一张图片提取的特征:")
    for key, value in attributes1.items():
        print(f"   {key}: {value}")
    print()
    
    # 处理第二张图片
    print("处理第二张图片...")
    result2 = extractor_low_conf.extract(image2_path, return_best_only=True)
    
    if result2 is None:
        print("  ⚠️  未检测到人物，使用整张图片进行分析")
        # 使用整张图片
        feature_output2 = feature_extractor.extract(img2_cv)
        attributes2 = feature_output2.attributes
        quality2 = 0.5  # 默认质量
        print(f"  ✅ VLM特征提取成功 ({len(attributes2)} 个属性)")
    else:
        print(f"  ✅ 检测到人物 (quality={result2.quality_score:.2f})")
        # 使用裁剪后的人物图片
        feature_output2 = feature_extractor.extract(result2.crop_image)
        attributes2 = feature_output2.attributes
        quality2 = result2.quality_score
        print(f"  ✅ VLM特征提取成功 ({len(attributes2)} 个属性)")
    
    if not feature_output2.is_valid:
        print("❌ 第二张图片特征提取失败")
        return
    
    # 显示第二张图片的属性
    print("\n📋 第二张图片提取的特征:")
    for key, value in attributes2.items():
        print(f"   {key}: {value}")
    print()
    
    # 比较分析
    print("=" * 50)
    print("🔍 特征比较分析:")
    
    # 比较共同属性
    common_attrs = set(attributes1.keys()) & set(attributes2.keys())
    matching_count = 0
    total_compared = 0
    
    if common_attrs:
        print(f"\n共同属性比较 ({len(common_attrs)} 个):")
        for attr in sorted(common_attrs):
            val1 = attributes1.get(attr, 'N/A')
            val2 = attributes2.get(attr, 'N/A')
            match = str(val1).lower() == str(val2).lower()
            total_compared += 1
            if match:
                matching_count += 1
            match_symbol = "✅" if match else "❌"
            print(f"   {match_symbol} {attr}: '{val1}' vs '{val2}'")
    
    # 计算匹配度
    if total_compared > 0:
        match_ratio = matching_count / total_compared
        print(f"\n📊 属性匹配度: {matching_count}/{total_compared} ({match_ratio:.1%})")
        
        # 关键属性权重更高
        key_attributes = ['gender', 'age_group', 'topwear', 'bottomwear', 'hair_style', 'glasses']
        key_match = 0
        key_total = 0
        
        for attr in key_attributes:
            if attr in attributes1 and attr in attributes2:
                key_total += 1
                if str(attributes1[attr]).lower() == str(attributes2[attr]).lower():
                    key_match += 1
        
        if key_total > 0:
            key_match_ratio = key_match / key_total
            print(f"🔑 关键属性匹配度: {key_match}/{key_total} ({key_match_ratio:.1%})")
            
            # 综合判断
            overall_score = 0.6 * match_ratio + 0.4 * key_match_ratio
            print(f"\n🎯 综合匹配分数: {overall_score:.3f}")
            
            threshold = 0.6
            if overall_score >= threshold:
                print(f"✅ 判断结果: 这两张图片中的人物很可能是同一个人")
                print(f"   (分数 {overall_score:.3f} >= 阈值 {threshold})")
            else:
                print(f"❌ 判断结果: 这两张图片中的人物可能不是同一个人")
                print(f"   (分数 {overall_score:.3f} < 阈值 {threshold})")
    
    print("\n" + "=" * 50)
    print("📊 总结:")
    print(f"图片1 质量分数: {quality1:.2f}")
    print(f"图片2 质量分数: {quality2:.2f}")
    print(f"图片1 属性数: {len(attributes1)}")
    print(f"图片2 属性数: {len(attributes2)}")

if __name__ == "__main__":
    compare_persons()
