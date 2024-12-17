from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

# Association table for user-domain relationships
user_domains = Table(
    'user_domains',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('domain_id', Integer, ForeignKey('domains.id'))
)

class Domain(Base):
    __tablename__ = "domains"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    is_active = Column(Boolean, default=True)
    data_sources = Column(String)  # JSON string of URLs

    # Relationships
    users = relationship("User", secondary=user_domains, back_populates="domains")
