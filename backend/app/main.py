from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
import logging
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import Session
from sqlalchemy import text
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database components in correct order to avoid circular imports
from app.db.base_class import Base  # noqa: F401
from app.db.session import engine, SessionLocal, init_db_connection
from app.api.v1.api import api_router

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with retry logic"""
    max_retries = 5
    retry_count = 0
    retry_interval = 5

    while retry_count < max_retries:
        try:
            logger.info(f"Attempting database initialization (attempt {retry_count + 1}/{max_retries})")
            # Test database connection
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("Database initialization successful")
            return True
        except (SQLAlchemyError, OperationalError) as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {str(e)}")
                return False
            logger.warning(f"Database initialization attempt {retry_count} failed, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    return False

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Starting application initialization...")
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Initialize database connection
        if not init_db_connection():
            logger.error("Failed to establish initial database connection")
            sys.exit(1)

        # Initialize database with required data
        from app.models.user import User
        from app.models.domain import Domain
        from app.core.security import get_password_hash
        db = SessionLocal()
        try:
            # Create admin user if it doesn't exist
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

            # Initialize domains if they don't exist
            domains = {
                "Technology": "Latest technology trends and innovations",
                "Finance": "Financial markets and business insights",
                "Medical": "Healthcare and medical advancements",
                "AI": "Artificial Intelligence and Machine Learning developments"
            }

            for name, description in domains.items():
                domain = db.query(Domain).filter(Domain.name == name).first()
                if not domain:
                    domain = Domain(
                        name=name,
                        description=description,
                        is_active=True,
                        data_sources="[]"
                    )
                    db.add(domain)
                    logger.info(f"Domain {name} created successfully")
            db.commit()
            logger.info("Domain initialization completed")

        finally:
            db.close()

        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

@app.get("/healthz")
async def healthz(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
