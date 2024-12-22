from pydantic import BaseModel

class Case(BaseModel):
    """案例模型"""
    content: str
    title: str = ""
    court: str = ""
    case_number: str = ""
    judgment_date: str = ""
