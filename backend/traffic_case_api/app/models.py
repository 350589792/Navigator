from pydantic import BaseModel
from typing import List

class CaseInput(BaseModel):
    case_text: str

class LawReference(BaseModel):
    law_name: str
    article_number: str
    content: str

class SimilarCase(BaseModel):
    title: str
    summary: str
    similarity_score: float

class CaseAnalysisResponse(BaseModel):
    relevant_laws: List[LawReference]
    similar_cases: List[SimilarCase]
