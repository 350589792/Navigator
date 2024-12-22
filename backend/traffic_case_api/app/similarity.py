from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .utils import custom_tokenizer

class SimilarityAnalyzer:
    def __init__(self, min_similarity: float = 0.15):
        """
        初始化相似度分析器
        
        Args:
            min_similarity: 最小相似度阈值，低于此值的案例将被过滤
        """
        self.vectorizer = TfidfVectorizer(
            tokenizer=None,  # 将在外部传入tokenizer
            preprocessor=None,
            token_pattern=None,  # 禁用默认的token模式，使用自定义tokenizer
            min_df=1,  # 至少出现1次的词才会被考虑
            max_df=0.9,  # 出现在90%以上文档中的词会被过滤掉
            # 为特征标记和数值标记设置更高的IDF权重
            # 动态词汇表，支持所有特征标记
            vocabulary=None,
            ngram_range=(1, 2),  # 支持单词和词组
            use_idf=True,  # 使用逆文档频率
            norm='l2',     # 使用L2归一化
            lowercase=True
        )
        self.min_similarity = min_similarity
        
    def fit_transform(self, texts: List[str], tokenizer=custom_tokenizer) -> np.ndarray:
        """
        对文本集合进行向量化转换
        
        Args:
            texts: 文本列表
            tokenizer: 自定义分词器函数
            
        Returns:
            文档-词项矩阵
        """
        self.vectorizer.tokenizer = tokenizer
        return self.vectorizer.fit_transform(texts)
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        对新的文本进行向量化转换
        
        Args:
            texts: 文本列表
            
        Returns:
            文档-词项矩阵
        """
        return self.vectorizer.transform(texts)
        
    def get_similar_cases(self, 
                         query_vector: np.ndarray, 
                         case_vectors: np.ndarray, 
                         cases: List[Dict],
                         top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        获取与查询案例最相似的案例
        
        Args:
            query_vector: 查询案例的向量
            case_vectors: 所有案例的向量矩阵
            cases: 案例详情列表
            top_k: 返回的相似案例数量
            
        Returns:
            List[Tuple[Dict, float]]: 相似案例及其相似度分数列表
        """
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, case_vectors)[0]
        
        # 确保相似度分数在有效范围内
        similarities = np.clip(similarities, 0, 1)
        
        # 过滤低于阈值的案例，但至少返回一个最相似的案例
        if len(similarities) > 0:
            max_sim = np.max(similarities)
            if max_sim >= self.min_similarity:
                filtered_indices = [i for i, sim in enumerate(similarities) if sim >= self.min_similarity]
            else:
                # 如果没有超过阈值的，返回最相似的一个
                filtered_indices = [np.argmax(similarities)]
            filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
        
        # 按相似度降序排序并获取top_k个结果
        similar_cases = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        # 返回案例详情和相似度分数
        return [(cases[idx], score) for idx, score in similar_cases]
