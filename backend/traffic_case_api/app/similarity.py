import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import custom_tokenizer, preprocess_text
from .models import SimilarCase

def get_similar_cases(input_text: str, all_cases: List[Dict], max_cases: int = 3) -> List[SimilarCase]:
    """
    获取与输入案例最相似的案例
    Args:
        input_text: 输入案例文本
        all_cases: 数据库中的所有案例
        max_cases: 返回的最大案例数量
    Returns:
        相似案例列表，按相似度降序排序
    """
    # 准备文本数据
    case_texts = [case["content"] for case in all_cases]
    case_texts.append(input_text)
    
    # 使用优化后的TF-IDF向量化
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        min_df=2,  # 忽略极少出现的词
        max_df=0.95,  # 忽略几乎所有文档都有的词
        sublinear_tf=True  # 使用对数缩放，减少长文本的权重偏差
    )
    tfidf_matrix = vectorizer.fit_transform(case_texts)
    
    # 计算相似度
    input_vector = tfidf_matrix[-1:]
    database_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(input_vector, database_vectors)[0]
    
    # 应用相似度阈值过滤
    threshold = 0.15
    filtered_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
    
    # 如果没有超过阈值的案例，适当降低阈值
    if not filtered_indices:
        threshold = 0.1
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
    
    # 获取最相似的案例
    similar_cases = []
    if filtered_indices:
        filtered_similarities = similarities[filtered_indices]
        top_k = min(max_cases, len(filtered_indices))
        top_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
        
        for idx_in_filtered in top_indices:
            original_idx = filtered_indices[idx_in_filtered]
            case = all_cases[original_idx]
            similar_cases.append(
                SimilarCase(
                    title=case.get("case_number", "未知案号"),
                    summary=case["content"][:200] + "..." if len(case["content"]) > 200 else case["content"],
                    similarity_score=float(similarities[original_idx])
                )
            )
    
    return similar_cases
