"""
阿里云DashScope API连接测试
"""

import sys

import numpy as np

from crossmedia_pid.config import DEFAULT_CONFIG_PATH, load_config
from crossmedia_pid.core.feature_vlm import create_feature_extractor


def create_test_image() -> np.ndarray:
    """创建一个测试图片（简单的人物轮廓）"""
    # 创建一个300x400的RGB图像
    img = np.ones((400, 300, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 画一个简单的"人物"（黑色矩形作为身体）
    img[100:350, 100:200] = [50, 50, 50]  # 身体
    img[50:120, 120:180] = [80, 60, 40]   # 头部
    
    return img


def test_aliyun_api():
    """测试阿里云API连接"""
    print("=" * 50)
    print("阿里云DashScope API连接测试")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    if not config:
        print(f"❌ 配置文件不存在: {DEFAULT_CONFIG_PATH}")
        return False
    
    vlm_config = config.get('models', {}).get('vlm', {})
    
    # 检查配置
    provider = vlm_config.get('provider')
    print(f"\n📋 配置信息:")
    print(f"   Provider: {provider}")
    print(f"   Model: {vlm_config.get('model_name')}")
    
    if provider != 'aliyun':
        print(f"\n⚠️  当前Provider是 '{provider}'，不是 'aliyun'")
        print("   请修改 configs/config.yaml 中的 provider 为 'aliyun'")
        return False
    
    api_key = vlm_config.get('api_key', '')
    if not api_key:
        print("\n❌ API Key未配置")
        print("   请设置 DASHSCOPE_API_KEY 环境变量或修改配置文件")
        return False
    
    # 隐藏部分API Key用于显示
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"   API Key: {masked_key}")
    
    # 创建测试图片
    print("\n🖼️  创建测试图片...")
    test_img = create_test_image()
    print(f"   图片尺寸: {test_img.shape}")
    
    # 初始化提取器
    print("\n🔧 初始化特征提取器...")
    try:
        extractor = create_feature_extractor(vlm_config)
        print("   ✓ 初始化成功")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return False
    
    # 测试API调用
    print("\n🚀 调用阿里云API...")
    print("   请稍候，首次调用可能需要10-30秒...")
    
    try:
        result = extractor.extract(test_img)
        
        print(f"\n📊 测试结果:")
        print(f"   状态: {'✓ 成功' if result.is_valid else '❌ 失败'}")
        print(f"   Provider: {result.provider}")
        
        if result.is_valid:
            print(f"\n✅ API连接测试成功！")
            print(f"\n📝 提取到的特征:")
            for key, value in result.attributes.items():
                print(f"   - {key}: {value}")
            
            print(f"\n🔍 原始响应 (前200字符):")
            print(f"   {result.raw_response[:200]}...")
            return True
        else:
            print(f"\n❌ API调用失败")
            print(f"   错误信息: {result.raw_response}")
            return False
            
    except Exception as e:
        print(f"\n❌ API调用异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """测试不同模型的可用性"""
    print("\n" + "=" * 50)
    print("测试不同模型配置")
    print("=" * 50)
    
    models_to_test = [
        ("qwen-vl-plus", "通义千问VL Plus"),
        ("qwen-vl-max", "通义千问VL Max"),
    ]
    
    # 加载配置获取API Key
    config = load_config()
    
    api_key = config.get('models', {}).get('vlm', {}).get('api_key', '')
    
    if not api_key or api_key.startswith('${'):
        print("❌ API Key未配置，跳过模型测试")
        return
    
    test_img = create_test_image()
    
    for model_id, model_name in models_to_test:
        print(f"\n🧪 测试 {model_name} ({model_id})...")
        
        try:
            extractor = create_feature_extractor({
                'provider': 'aliyun',
                'api_key': api_key,
                'model_name': model_id,
                'max_tokens': 256,
                'temperature': 0.1
            })
            
            result = extractor.extract(test_img)
            
            if result.is_valid:
                print(f"   ✓ {model_name} 可用")
                print(f"     提取特征数: {len(result.attributes)}")
            else:
                print(f"   ✗ {model_name} 调用失败")
                print(f"     错误: {result.raw_response[:100]}")
                
        except Exception as e:
            print(f"   ✗ {model_name} 异常: {str(e)[:100]}")


if __name__ == '__main__':
    # 主测试
    success = test_aliyun_api()
    
    if success:
        # 如果主测试成功，测试其他模型
        test_models()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ 测试失败，请检查配置")
        print("=" * 50)
        sys.exit(1)
