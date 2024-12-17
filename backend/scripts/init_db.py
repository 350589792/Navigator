import os
import sys
from pathlib import Path
import logging
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.session import SessionLocal, engine
from app.db.base_class import Base
from app.models.user import User
from app.core.security import get_password_hash

def init_db():
    try:
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Initialize database connection
        db = SessionLocal()
        try:
            # Test connection
            logger.info("Testing database connection...")
            db.execute(text("SELECT 1"))
            logger.info("Database connection test successful")

            # Check if admin user exists
            logger.info("Checking for admin user...")
            admin = db.query(User).filter(User.email == "admin@example.com").first()
            if not admin:
                admin = User(
                    email="admin@example.com",
                    hashed_password=get_password_hash("admin123"),
                    is_admin=True,
                    is_active=True
                )
                db.add(admin)
                db.commit()
                logger.info("Admin user created successfully")
            else:
                logger.info("Admin user already exists")
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    init_db()
    logger.info("Database initialization completed successfully")
