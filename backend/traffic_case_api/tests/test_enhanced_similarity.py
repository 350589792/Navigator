import pytest
import numpy as np
from app.similarity import SimilarityAnalyzer
from app.utils import custom_tokenizer

def test_keyword_weighting():
    """测试关键词权重是否正确应用"""
    analyzer = SimilarityAnalyzer()
    cases = [
        "一起普通的交通事故案件",
        "一起严重的酒驾肇事案件",
        "一起普通的超速案件"
    ]
    
    # 转换文本
    vectors = analyzer.fit_transform(cases)
    
    # 获取特征名称
    feature_names = analyzer.vectorizer.get_feature_names_out()
    
    # 检查"酒驾"的权重是否被加强
    if '酒驾' in feature_names:
        drunk_driving_idx = feature_names.tolist().index('酒驾')
        drunk_driving_weights = vectors[:, drunk_driving_idx].toarray().flatten()
        
        # 第二个案例（酒驾案件）的权重应该明显高于其他案例
        assert drunk_driving_weights[1] > drunk_driving_weights[0]
        assert drunk_driving_weights[1] > drunk_driving_weights[2]

def test_similar_case_matching():
    """测试相似案例匹配的准确性"""
    analyzer = SimilarityAnalyzer()
    cases = [
        "张某酒后驾驶机动车发生交通事故，造成一人受伤",
        "李某醉酒驾驶汽车，与行人发生碰撞致其重伤",
        "王某超速行驶撞伤路人",
        "赵某驾驶机动车闯红灯撞伤他人"
    ]
    
    # 将查询和案例一起进行fit_transform以确保共享相同的特征空间
    all_texts = cases + ["陈某饮酒后驾驶汽车发生事故"]
    all_vectors = analyzer.fit_transform(all_texts)
    
    # 分离案例向量和查询向量
    case_vectors = all_vectors[:-1]  # 除最后一个向量外的所有向量是案例
    query_vector = all_vectors[-1:].reshape(1, -1)  # 最后一个向量是查询，确保形状正确
    
    similar_cases = analyzer.get_similar_cases(
        query_vector, 
        case_vectors,
        [{"content": c} for c in cases],
        top_k=2
    )
    
    # 验证返回的相似案例是否包含酒驾相关案例
    assert len(similar_cases) > 0
    assert any('酒' in case[0]['content'] or '醉' in case[0]['content'] 
              for case in similar_cases)
    
    # 验证相似度分数
    assert all(score >= 0.30 for _, score in similar_cases)

def test_multiple_keyword_matching():
    """测试多个关键词的匹配效果"""
    analyzer = SimilarityAnalyzer()
    cases = [
        "张某酒后驾驶机动车超速行驶，造成一人死亡",
        "李某醉酒驾驶汽车，与行人发生碰撞",
        "王某驾驶机动车闯红灯撞伤他人",
        "赵某驾驶机动车逃逸致人重伤"
    ]
    
    # 将查询和案例一起进行fit_transform以确保共享相同的特征空间
    all_texts = cases + ["黄某酒后驾驶并超速行驶发生事故"]
    all_vectors = analyzer.fit_transform(all_texts)
    
    # 分离案例向量和查询向量
    case_vectors = all_vectors[:-1]  # 除最后一个向量外的所有向量是案例
    query_vector = all_vectors[-1:].reshape(1, -1)  # 最后一个向量是查询，确保形状正确
    
    similar_cases = analyzer.get_similar_cases(
        query_vector,
        case_vectors,
        [{"content": c} for c in cases],
        top_k=2
    )
    
    # 验证最相似的案例应该是第一个（包含酒驾和超速）
    assert similar_cases[0][0]['content'] == cases[0]
    assert similar_cases[0][1] >= 0.30  # 相似度应该高于阈值

def test_ngram_effectiveness():
    """测试n-gram特征的有效性"""
    analyzer = SimilarityAnalyzer()
    cases = [
        "张某酒后驾驶机动车",
        "李某驾驶机动车",
        "王某酒后驾驶"
    ]
    
    # 将查询和案例一起进行fit_transform以确保共享相同的特征空间
    all_texts = cases + ["酒后驾驶机动车"]
    all_vectors = analyzer.fit_transform(all_texts)
    feature_names = analyzer.vectorizer.get_feature_names_out()
    
    # 验证是否生成了bigram和trigram特征
    assert any(len(feature.split()) > 1 for feature in feature_names)
    
    # 分离案例向量和查询向量
    vectors = all_vectors[:-1]  # 除最后一个向量外的所有向量是案例
    query_vector = all_vectors[-1:].reshape(1, -1)  # 最后一个向量是查询，确保形状正确
    
    similar_cases = analyzer.get_similar_cases(
        query_vector,
        vectors,
        [{"content": c} for c in cases],
        top_k=2
    )
    
    # 验证最相似的应该是完整匹配的案例
    assert similar_cases[0][0]['content'] == cases[0]
