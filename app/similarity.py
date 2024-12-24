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
        # 定义重要法律关键词及其权重，提高酒驾相关词的权重
        self.legal_keywords = {
            # 酒驾相关词汇，权重最高
            '酒驾': 5.0,
            '醉驾': 5.0,
            '酒后驾驶': 5.0,
            '醉酒驾驶': 5.0,
            '饮酒驾驶': 5.0,
            '酒后': 4.0,
            '醉酒': 4.0,
            '饮酒': 4.0,
            
            # 其他严重违法行为
            '超速': 4.5,
            '超速行驶': 4.5,
            '速度过快': 4.5,
            '高速': 3.5,
            '逃逸': 2.0,
            '肇事逃逸': 2.0,
            '闯红灯': 2.0,
            
            # 事故后果
            '死亡': 2.5,
            '重伤': 2.5,
            '伤亡': 2.0,
            '受伤': 2.0,
            
            # 一般违法行为
            '肇事': 1.5,
            '违章': 1.5,
            '违法': 1.5
        }
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=None,  # 将在外部传入tokenizer
            preprocessor=None,
            token_pattern=None,  # 禁用默认的token模式，使用自定义tokenizer
            min_df=1,  # 允许所有词，因为我们的案例可能包含重要但罕见的词
            max_df=0.95,  # 允许更多常见词，因为法律术语经常重复
            vocabulary=None,
            ngram_range=(1, 4),  # 支持更长的词组，最多4个词的组合以捕获更多语义
            use_idf=True,
            norm='l2',
            lowercase=True
        )
        self.min_similarity = min_similarity
        
    def fit_transform(self, texts: List[str], tokenizer=custom_tokenizer) -> np.ndarray:
        """
        对文本集合进行向量化转换，并应用关键词权重
        
        Args:
            texts: 文本列表
            tokenizer: 自定义分词器函数
            
        Returns:
            文档-词项矩阵
        """
        self.vectorizer.tokenizer = tokenizer
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # 获取特征名称（词汇表）
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 应用关键词权重
        for keyword, weight in self.legal_keywords.items():
            if keyword in feature_names:
                keyword_idx = feature_names.tolist().index(keyword)
                tfidf_matrix.data[tfidf_matrix.indices == keyword_idx] *= weight
        
        return tfidf_matrix
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        对新的文本进行向量化转换，并应用关键词权重
        
        Args:
            texts: 文本列表
            
        Returns:
            文档-词项矩阵
        """
        # 将查询文本添加到现有的案例向量中
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # 获取特征名称（词汇表）
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 应用关键词权重，包括复合词
        for keyword, weight in self.legal_keywords.items():
            # 处理复合词
            keyword_parts = keyword.split()
            for i in range(len(keyword_parts)):
                for j in range(i + 1, len(keyword_parts) + 1):
                    sub_keyword = " ".join(keyword_parts[i:j])
                    if sub_keyword in feature_names:
                        keyword_idx = feature_names.tolist().index(sub_keyword)
                        # 对于部分匹配，使用较小的权重
                        part_weight = weight * (j - i) / len(keyword_parts)
                        tfidf_matrix.data[tfidf_matrix.indices == keyword_idx] *= part_weight
        
        return tfidf_matrix
        
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
        # 计算基础余弦相似度
        base_similarities = cosine_similarity(query_vector, case_vectors)[0]
        
        # 从向量重建查询文本
        query_features = query_vector.nonzero()[1]
        query_text = " ".join(self.vectorizer.get_feature_names_out()[i] for i in query_features)
        query_keywords = [kw for kw in self.legal_keywords.keys() if kw in query_text]
        query_weight = sum(self.legal_keywords[kw] for kw in query_keywords) if query_keywords else 1.0
        
        # 计算每个案例的关键词匹配分数
        keyword_similarities = np.zeros_like(base_similarities)
        for idx, case_dict in enumerate(cases):
            case_text = case_dict['content']
            case_keywords = set()
            # 检查完整关键词和部分匹配
            for keyword in self.legal_keywords.keys():
                if keyword in case_text:
                    case_keywords.add(keyword)
                else:
                    # 检查关键词的部分匹配
                    parts = keyword.split()
                    if len(parts) > 1 and all(part in case_text for part in parts):
                        case_keywords.add(keyword)
            
            if case_keywords and query_keywords:
                # 计算关键词匹配度
                common_keywords = case_keywords & set(query_keywords)
                if common_keywords:
                    # 基于共同关键词计算相似度，给予更高权重
                    common_weight = sum(self.legal_keywords[kw] for kw in common_keywords)
                    max_weight = max(self.legal_keywords[kw] for kw in common_keywords)
                    # 使用最大权重和累积权重的组合，并提高权重
                    keyword_similarities[idx] = (max_weight * 0.7 + (common_weight / len(common_keywords)) * 0.5)
                    # 如果包含重要关键词（如酒驾、醉驾等），给予额外加分
                    important_keywords = {'酒后驾驶', '醉酒驾驶', '酒驾', '醉驾'}
                    if any(kw in important_keywords for kw in common_keywords):
                        keyword_similarities[idx] *= 1.3  # 重要关键词匹配时提升30%的分数
        
        # 组合相似度分数，进一步提高关键词匹配的权重
        # 对于关键词匹配分数高的案例（>0.5），给予额外的权重提升
        boost_mask = keyword_similarities > 0.5
        keyword_similarities[boost_mask] *= 1.2  # 提升高匹配度的关键词得分
        
        # 动态调整权重：根据关键词匹配程度调整权重比例
        weight_mask = keyword_similarities > 0.5
        base_weights = np.where(weight_mask, 0.05, 0.2)  # 高匹配度时基础相似度权重更低
        keyword_weights = np.where(weight_mask, 0.95, 0.8)  # 高匹配度时关键词权重更高
        combined_similarities = base_similarities * base_weights + keyword_similarities * keyword_weights
        
        # 确保相似度分数在有效范围内
        combined_similarities = np.clip(combined_similarities, 0, 1)
        
        # 过滤低于阈值的案例，但至少返回一个最相似的案例
        if len(combined_similarities) > 0:
            max_sim = np.max(combined_similarities)
            if max_sim >= self.min_similarity:
                filtered_indices = [i for i, sim in enumerate(combined_similarities) 
                                 if sim >= self.min_similarity]
            else:
                # 如果没有超过阈值的，返回最相似的一个
                filtered_indices = [np.argmax(combined_similarities)]
            filtered_similarities = [(i, combined_similarities[i]) for i in filtered_indices]
        
        # 按相似度降序排序并获取top_k个结果
        similar_cases = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        # 返回案例详情和相似度分数
        return [(cases[idx], score) for idx, score in similar_cases]
