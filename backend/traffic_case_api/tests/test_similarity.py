import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.similarity import get_similar_cases
from app.utils import custom_tokenizer, preprocess_text

# 测试案例数据
TEST_CASES = [
    {
        "title": "超速案例1",
        "content": "被告人驾驶机动车在高速公路上以140公里/小时的速度行驶，超过规定时速120公里/小时的限制。造成一起交通事故。"
    },
    {
        "title": "超速案例2",
        "content": "被告人驾驶小型轿车，在限速80公里/小时的道路上以135公里/小时的速度行驶。"
    },
    {
        "title": "闯红灯案例",
        "content": "被告人驾驶机动车闯红灯，与正常通行的行人发生碰撞，导致一人受伤。事故发生后，被告人立即下车救助伤者。"
    },
    {
        "title": "酒驾案例1",
        "content": "被告人血液中乙醇含量为135.6mg/100ml，属于醉酒驾驶机动车。造成与对向车辆相撞。"
    },
    {
        "title": "酒驾案例2",
        "content": "被告人饮酒后驾驶机动车，血液检测结果为98.5mg/100ml，达到醉驾标准。"
    }
]

def test_speed_case_matching():
    """测试超速案例匹配"""
    input_text = "驾驶员在高速公路上以138公里/小时的速度行驶"
    similar_cases = get_similar_cases(input_text, TEST_CASES)
    
    assert len(similar_cases) > 0
    # 验证是否匹配到超速案例
    assert any("140公里/小时" in case.summary for case in similar_cases)
    assert any("135公里/小时" in case.summary for case in similar_cases)
    # 验证相似度分数
    assert max(case.similarity_score for case in similar_cases) > 0.15

def test_traffic_signal_violation():
    """测试交通信号违规检测"""
    input_text = "驾驶员闯红灯撞伤行人，事后及时救助"
    similar_cases = get_similar_cases(input_text, TEST_CASES)
    
    assert len(similar_cases) > 0
    # 验证是否匹配到闯红灯案例
    matched_case = next((case for case in similar_cases if "闯红灯" in case.summary), None)
    assert matched_case is not None
    assert matched_case.similarity_score > 0.15
    assert "行人" in matched_case.summary
    assert "救助" in matched_case.summary

def test_drunk_driving_detection():
    """测试酒驾特征检测"""
    input_text = "驾驶员血液酒精含量为102.5mg/100ml，造成交通事故"
    similar_cases = get_similar_cases(input_text, TEST_CASES)
    
    assert len(similar_cases) > 0
    # 验证是否匹配到酒驾案例
    drunk_cases = [case for case in similar_cases if "醉" in case.summary]
    assert len(drunk_cases) >= 1
    # 验证相似度分数
    assert max(case.similarity_score for case in drunk_cases) > 0.15

def test_multiple_feature_detection():
    """测试多特征检测"""
    input_text = "驾驶员饮酒后以130公里/小时的速度行驶，并闯红灯"
    similar_cases = get_similar_cases(input_text, TEST_CASES)
    
    assert len(similar_cases) >= 2
    # 验证是否匹配到多种类型的案例
    features_found = {
        'speed': False,
        'drunk': False,
        'signal': False
    }
    
    for case in similar_cases:
        if any(speed in case.summary for speed in ["140公里/小时", "135公里/小时"]):
            features_found['speed'] = True
        if "醉" in case.summary or "酒" in case.summary:
            features_found['drunk'] = True
        if "闯红灯" in case.summary:
            features_found['signal'] = True
    
    # 验证至少匹配到两种特征
    assert sum(features_found.values()) >= 2

def test_api_case_analysis(test_client):
    """测试API案例分析端点"""
    # 确保测试数据已经准备好
    test_case = {
        "case_text": "驾驶员血液酒精含量为102.5mg/100ml，以135公里/小时的速度行驶"
    }
    
    response = test_client.post(
        "/analyze_case",
        json=test_case
    )
    
    # 验证响应状态码和结构
    assert response.status_code == 200
    data = response.json()
    
    # 验证响应包含所需字段
    assert "relevant_laws" in data
    assert "similar_cases" in data
    
    # 验证法条列表不为空
    assert len(data["relevant_laws"]) > 0
    
    # 验证每个法条的结构
    for law in data["relevant_laws"]:
        assert "law_name" in law
        assert "article_number" in law
        assert "content" in law
    
    # 验证相似案例列表不为空
    assert len(data["similar_cases"]) > 0
    
    # 验证每个相似案例的结构
    for case in data["similar_cases"]:
        assert "title" in case
        assert "summary" in case
        assert "similarity_score" in case
        assert case["similarity_score"] >= 0.0
        assert case["similarity_score"] <= 1.0
    
    assert response.status_code == 200
    data = response.json()
    
    # 验证响应结构
    assert "similar_cases" in data
    assert "relevant_laws" in data
    
    # 验证相似案例
    assert len(data["similar_cases"]) > 0
    # 验证至少一个案例相似度大于阈值
    assert any(case["similarity_score"] > 0.15 for case in data["similar_cases"])
    # 验证法律条款
    assert len(data["relevant_laws"]) > 0

def test_tokenizer_features():
    """测试分词器特征标记"""
    text = "驾驶员血液酒精含量为102.5mg/100ml，以135公里/小时的速度行驶"
    tokens = custom_tokenizer(text)
    
    # 验证速度特征标记
    speed_tokens = [t for t in tokens if "SPEED" in t]
    assert len(speed_tokens) > 0
    assert any("135" in t for t in speed_tokens)
    
    # 验证酒驾特征标记
    bac_tokens = [t for t in tokens if "BAC" in t]
    assert len(bac_tokens) > 0
    assert any("102.5" in t for t in bac_tokens)

def test_text_preprocessing():
    """测试文本预处理"""
    text = "驾驶员血液酒精含量为102.5mg/100ml，以135公里/小时的速度行驶"
    processed_text = preprocess_text(text)
    
    # 验证预处理后的文本包含特征标记
    assert "SPEED" in processed_text
    assert "BAC" in processed_text
    assert "102.5" in processed_text
    assert "135" in processed_text
