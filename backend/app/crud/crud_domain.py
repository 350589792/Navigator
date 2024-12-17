from typing import List, Optional
from sqlalchemy.orm import Session
from app.crud.base import CRUDBase
from app.models.domain import Domain
from app.schemas.domain import DomainCreate, DomainUpdate

class CRUDDomain(CRUDBase[Domain, DomainCreate, DomainUpdate]):
    def get_by_name(self, db: Session, *, name: str) -> Optional[Domain]:
        return db.query(self.model).filter(self.model.name == name).first()

    def get_multi_by_user(
        self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[Domain]:
        return (
            db.query(self.model)
            .join(self.model.users)
            .filter(self.model.users.any(id=user_id))
            .offset(skip)
            .limit(limit)
            .all()
        )

crud_domain = CRUDDomain(Domain)
