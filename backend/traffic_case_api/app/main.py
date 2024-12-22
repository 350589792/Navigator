from fastapi import FastAPI, HTTPException
from .models import Case
from .data_manager import CaseDataManager
from .similarity import SimilarityAnalyzer
from .utils import custom_tokenizer

app = FastAPI()
data_manager = CaseDataManager()
similarity_analyzer = SimilarityAnalyzer(min_similarity=0.15)

@app.post("/analyze_case")
async def analyze_case(case: Case):
    try:
        # 获取所有案例
        all_cases = await data_manager.get_all_cases()
        if not all_cases:
            raise HTTPException(status_code=404, detail="No cases found in database")
            
        # 准备文本数据
        case_texts = [c["content"] for c in all_cases]
        
        # 向量化现有案例
        case_vectors = similarity_analyzer.fit_transform(case_texts, custom_tokenizer)
        
        # 向量化查询案例
        query_vector = similarity_analyzer.transform([case.content])
        
        # 获取相似案例
        similar_cases_with_scores = similarity_analyzer.get_similar_cases(
            query_vector=query_vector,
            case_vectors=case_vectors,
            cases=all_cases
        )
        
        # 格式化返回结果
        similar_cases = [
            {
                **case_info,
                "similarity_score": float(score)
            }
            for case_info, score in similar_cases_with_scores
        ]
        
        # 根据案例特征选择相关法条
        relevant_laws = ["《中华人民共和国道路交通安全法》第一百一十九条"]
        if "酒后" in case.content or "醉酒" in case.content:
            relevant_laws.append("《中华人民共和国刑法》第一百三十三条")
        if "逃逸" in case.content:
            relevant_laws.append("《中华人民共和国刑法》第一百三十三条之一")
        
        return {
            "similar_cases": similar_cases,
            "relevant_laws": relevant_laws
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
