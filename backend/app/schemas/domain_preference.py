from pydantic import BaseModel
from typing import List

class DomainPreferenceBase(BaseModel):
    domains: List[str]

    class Config:
        from_attributes = True

class DomainPreferenceCreate(DomainPreferenceBase):
    pass

class DomainPreferenceUpdate(DomainPreferenceBase):
    pass

class DomainPreference(DomainPreferenceBase):
    id: int | None = None
    user_id: int | None = None
