from pydantic import BaseModel
from typing import List, Optional

class DomainBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_sources: Optional[str] = None
    is_active: bool = True

class DomainCreate(DomainBase):
    pass

class DomainUpdate(DomainBase):
    pass

class Domain(DomainBase):
    id: int

    class Config:
        from_attributes = True

class DomainInDB(Domain):
    pass
