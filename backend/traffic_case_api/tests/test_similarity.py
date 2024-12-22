import pytest
import numpy as np
from app.similarity import SimilarityAnalyzer
from app.utils import custom_tokenizer

def test_similarity_threshold():
    """测试相似度阈值过滤"""
    analyzer = SimilarityAnalyzer(min_similarity=0.15)
    
    cases = [
        "驾驶员甲在高速公路上以120公里/小时的速度行驶，造成追尾事故",
        "驾驶员乙酒后驾驶机动车，在城市道路上发生碰撞",
        "驾驶员丙闯红灯，与一辆正常行驶的车辆相撞后逃逸"
    ]
    
    query = "驾驶员在高速公路超速行驶致追尾"
    
    # 向量化案例
    case_vectors = analyzer.fit_transform(cases, custom_tokenizer)
    query_vector = analyzer.transform([query])
    
    # 获取相似案例
    similar_cases = analyzer.get_similar_cases(
        query_vector=query_vector,
        case_vectors=case_vectors,
        cases=[{"content": c} for c in cases]
    )
    
    # 验证结果
    assert len(similar_cases) > 0
    assert similar_cases[0][1] >= 0.15  # 相似度应该高于阈值
    assert "120公里/小时" in similar_cases[0][0]["content"]  # 应该匹配到超速案例

def test_feature_weighting():
    """测试特征权重"""
    analyzer = SimilarityAnalyzer(min_similarity=0.15)
    
    cases = [
        "驾驶员在普通道路上行驶，未注意前方车辆",
        "驾驶员酒后驾驶，血液中酒精含量为80mg/100ml",
        "驾驶员闯红灯撞向行人"
    ]
    
    query = "驾驶员饮酒后驾驶机动车"
    
    # 向量化案例
    case_vectors = analyzer.fit_transform(cases, custom_tokenizer)
    query_vector = analyzer.transform([query])
    
    # 获取相似案例
    similar_cases = analyzer.get_similar_cases(
        query_vector=query_vector,
        case_vectors=case_vectors,
        cases=[{"content": c} for c in cases]
    )
    
    # 验证结果
    assert "酒后" in similar_cases[0][0]["content"]  # 应该优先匹配酒驾案例
    assert similar_cases[0][1] > 0.2  # 特征标记应该提供更高的相似度分数
