# Import all endpoint modules
from . import auth
from . import users
from . import domains
from . import reports
from . import admin

# Make them available for import
__all__ = ["auth", "users", "domains", "reports", "admin"]
