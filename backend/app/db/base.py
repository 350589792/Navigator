# Import all models here for Alembic autogenerate support
from app.db.base_class import Base  # noqa: F401

# Import all models
from app.models.user import User  # noqa: F401
from app.models.domain import Domain  # noqa: F401
from app.models.report import Report  # noqa: F401

# Make them available for import
__all__ = ["Base", "User", "Domain", "Report"]
