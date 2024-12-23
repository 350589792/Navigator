from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import Case
from .data_manager import CaseDataManager
from .similarity import SimilarityAnalyzer
from .utils import custom_tokenizer

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://*.devin.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
data_manager = CaseDataManager()
similarity_analyzer = SimilarityAnalyzer(min_similarity=0.15)

@app.post("/analyze_case")
async def analyze_case(case: Case):
    # 验证输入
    if not case.content or len(case.content.strip()) == 0:
        raise HTTPException(status_code=422, detail="Case content cannot be empty")
    
    # 获取所有案例
    all_cases = await data_manager.get_all_cases()
    if not all_cases:
        raise HTTPException(status_code=404, detail="No cases found in database")
        
    try:
        # 准备文本数据
        case_texts = [c["content"] for c in all_cases]
        
        # 向量化现有案例
        case_vectors = similarity_analyzer.fit_transform(case_texts, custom_tokenizer)
        
        # 向量化查询案例
        query_vector = similarity_analyzer.transform([case.content])
        
        # 获取相似案例 - 提高相似度阈值
        similar_cases_with_scores = similarity_analyzer.get_similar_cases(
            query_vector=query_vector,
            case_vectors=case_vectors,
            cases=all_cases,
            top_k=5  # 限制返回最相关的5个案例
        )
        
        # 格式化返回结果
        similar_cases = [
            {
                **case_info,
                "similarity_score": float(score)
            }
            for case_info, score in similar_cases_with_scores
            if float(score) > 0.3  # 只返回相似度大于30%的案例
        ]
        
        # 根据案例特征选择相关法条
        relevant_laws = []
        
        # 基础交通违法
        relevant_laws.append("《中华人民共和国道路交通安全法》第一百一十九条")
        
        # 酒驾醉驾情况
        if any(keyword in case.content for keyword in ["酒后", "醉酒", "饮酒", "酒驾"]):
            relevant_laws.append("《中华人民共和国刑法》第一百三十三条")
            relevant_laws.append("《中华人民共和国道路交通安全法》第九十一条")
        
        # 肇事逃逸
        if any(keyword in case.content for keyword in ["逃逸", "逃离", "离开现场"]):
            relevant_laws.append("《中华人民共和国刑法》第一百三十三条之一")
            relevant_laws.append("《中华人民共和国道路交通安全法》第七十条")
            
        # 超速行驶
        if any(keyword in case.content for keyword in ["超速", "高速", "超过规定时速"]):
            relevant_laws.append("《中华人民共和国道路交通安全法》第四十二条")
            
        # 违章驾驶
        if any(keyword in case.content for keyword in ["违章", "违规", "闯红灯"]):
            relevant_laws.append("《中华人民共和国道路交通安全法》第九十条")
            
        # 无证驾驶
        if any(keyword in case.content for keyword in ["无证", "无驾驶证", "无有效驾驶证"]):
            relevant_laws.append("《中华人民共和国道路交通安全法》第九十九条")
        
        return {
            "similar_cases": similar_cases,
            "relevant_laws": relevant_laws
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
