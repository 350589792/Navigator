from .crud_user import crud_user
from .crud_domain import crud_domain
from .crud_report import crud_report
from .base import CRUDBase

__all__ = ["CRUDBase", "crud_user", "crud_domain", "crud_report"]
