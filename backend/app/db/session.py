from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from app.core.config import settings
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_engine():
    return create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        connect_args={"connect_timeout": 10} if not settings.DATABASE_URL.startswith("sqlite") else {"check_same_thread": False}
    )

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db_connection(max_retries=5, retry_interval=5):
    retry_count = 0
    last_exception = None

    while retry_count < max_retries:
        try:
            logger.info(f"Attempting database connection (attempt {retry_count + 1}/{max_retries})")
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("Database connection successful")
            return True
        except (SQLAlchemyError, OperationalError) as e:
            last_exception = e
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                break
            logger.warning(f"Database connection attempt {retry_count} failed, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    if last_exception:
        raise last_exception
    return False
