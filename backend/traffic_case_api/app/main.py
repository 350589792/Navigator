from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import CaseInput, CaseAnalysisResponse, LawReference, SimilarCase
from .data_manager import CaseDataManager
from .scraper import TrafficCaseScraper

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize data manager and scraper
data_manager = None
case_scraper = None

# Initialize the database with real cases if empty
@app.on_event("startup")
async def initialize_database():
    global data_manager, case_scraper
    data_manager = await CaseDataManager.create()
    case_scraper = await TrafficCaseScraper.create(test_mode=True)
    cases = await data_manager.get_all_cases()
    if not cases:
        await case_scraper.update_case_database()

def preprocess_text(text: str) -> str:
    """对中文文本进行预处理"""
    words = jieba.cut(text)
    return " ".join(words)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/analyze_case", response_model=CaseAnalysisResponse)
async def analyze_case(case_input: CaseInput, background_tasks: BackgroundTasks):
    """分析交通事故案例，返回相关法条和相似案例"""
    # Get all cases from the database
    all_cases = await data_manager.get_all_cases()
    
    # Process the input case
    case_texts = [case["content"] for case in all_cases]
    case_texts.append(case_input.case_text)
    
    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer(tokenizer=lambda x: jieba.lcut(x))
    tfidf_matrix = vectorizer.fit_transform(case_texts)
    
    # Calculate similarity between input case and all other cases
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # Get top 3 similar cases
    similar_cases = []
    top_indices = np.argsort(similarities)[-3:][::-1]
    for idx in top_indices:
        case = all_cases[idx]
        similar_cases.append(
            SimilarCase(
                title=case["title"],
                summary=case["content"][:200] + "..." if len(case["content"]) > 200 else case["content"],
                similarity_score=float(similarities[idx])
            )
        )
    
    # Get relevant laws from the database
    all_laws = await data_manager.get_all_laws()
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
    background_tasks.add_task(case_scraper.update_case_database)
    
    return CaseAnalysisResponse(
        relevant_laws=relevant_laws,
        similar_cases=similar_cases
    )
