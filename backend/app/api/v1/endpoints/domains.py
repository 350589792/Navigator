from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.api import deps
from app.crud import crud_domain
from app.schemas.domain import Domain, DomainCreate, DomainUpdate
from app.models.user import User

router = APIRouter()

@router.get("/", response_model=List[Domain])
def get_domains(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Get all domains.
    """
    if current_user.is_admin:
        return crud_domain.get_multi(db)
    return current_user.domains

@router.post("/", response_model=Domain)
def create_domain(
    *,
    db: Session = Depends(deps.get_db),
    domain_in: DomainCreate,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Create new domain. Admin only.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_domain.create(db, obj_in=domain_in)

@router.put("/{domain_id}", response_model=Domain)
def update_domain(
    *,
    db: Session = Depends(deps.get_db),
    domain_id: int,
    domain_in: DomainUpdate,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Update domain. Admin only.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    domain = crud_domain.get(db, id=domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    return crud_domain.update(db, db_obj=domain, obj_in=domain_in)

@router.post("/preferences")
def update_user_domains(
    *,
    db: Session = Depends(deps.get_db),
    domain_ids: List[int],
    current_user: User = Depends(deps.get_current_user)
):
    """
    Update user's domain preferences.
    """
    domains = []
    for domain_id in domain_ids:
        domain = crud_domain.get(db, id=domain_id)
        if not domain:
            raise HTTPException(
                status_code=404,
                detail=f"Domain with id {domain_id} not found"
            )
        domains.append(domain)

    current_user.domains = domains
    db.commit()
    return {"status": "success"}
