from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ReportBase(BaseModel):
    title: str
    content: str
    summary: Optional[str] = None

class ReportCreate(ReportBase):
    user_id: int

class ReportUpdate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class ReportInDB(Report):
    pass
