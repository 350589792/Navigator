from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import CaseInput, CaseAnalysisResponse, LawReference, SimilarCase
from .data_manager import CaseDataManager
from .scraper import TrafficCaseScraper
from .similarity import get_similar_cases  # Import similarity calculation module

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """处理应用程序的生命周期事件"""
    # 启动时初始化
    app.state.data_manager = await CaseDataManager.create()
    app.state.case_scraper = await TrafficCaseScraper.create(test_mode=True)
    cases = await app.state.data_manager.get_all_cases()
    if not cases:
        await app.state.case_scraper.update_case_database()
    yield
    # 关闭时清理
    if hasattr(app.state, 'data_manager'):
        await app.state.data_manager.clear_database()
    app.state.data_manager = None
    app.state.case_scraper = None

app = FastAPI(lifespan=lifespan)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/analyze_case", response_model=CaseAnalysisResponse)
async def analyze_case(case_input: CaseInput, background_tasks: BackgroundTasks):
    """分析交通事故案例，返回相关法条和相似案例"""
    # Get all cases from the database
    all_cases = await app.state.data_manager.get_all_cases()
    
    # Process the input case
    case_texts = [case["content"] for case in all_cases]
    case_texts.append(case_input.case_text)
    
    # 使用相似度计算模块获取相似案例
    similar_cases = get_similar_cases(case_input.case_text, all_cases)
    
    # Get relevant laws from the database
    all_laws = await app.state.data_manager.get_all_laws()
    relevant_laws = []
    for category in all_laws:
        for law in all_laws[category]:
            relevant_laws.append(
                LawReference(
                    law_name=law["law_name"],
                    article_number=law["article_number"],
                    content=law["content"]
                )
            )
    
    # Schedule background task to update cases
    background_tasks.add_task(app.state.case_scraper.update_case_database)
    
    return CaseAnalysisResponse(
        relevant_laws=relevant_laws,
        similar_cases=similar_cases
    )
