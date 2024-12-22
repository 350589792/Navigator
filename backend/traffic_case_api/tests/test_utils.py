import pytest
from app.utils import add_feature_tokens, custom_tokenizer

def test_feature_detection():
    # 测试超速特征
    text = "驾驶员以120公里/小时的速度行驶"
    result = add_feature_tokens(text)
    assert "FEATURE_公里/小时" in result
    
    # 测试肇事逃逸特征
    text = "驾驶员肇事逃逸"
    result = add_feature_tokens(text)
    assert "FEATURE_肇事逃逸" in result
    
    # 测试交通信号违规
    text = "驾驶员闯红灯导致事故"
    result = add_feature_tokens(text)
    assert "FEATURE_闯红灯" in result
    
    # 测试车辆故障
    text = "因车辆制动失灵导致追尾"
    result = add_feature_tokens(text)
    assert "FEATURE_制动失灵" in result
    
    # 测试行人事故
    text = "在人行横道撞击行人"
    result = add_feature_tokens(text)
    assert "FEATURE_行人" in result
    assert "FEATURE_人行横道" in result

def test_custom_tokenizer():
    # 测试分词结果包含特征标记
    text = "驾驶员超速行驶并闯红灯"
    tokens = custom_tokenizer(text)
    
    # 验证基础分词
    assert "驾驶员" in tokens
    assert "超速" in tokens
    assert "行驶" in tokens
    
    # 验证特征标记
    assert any("FEATURE_超速" in token for token in tokens)
    assert any("FEATURE_闯红灯" in token for token in tokens)
