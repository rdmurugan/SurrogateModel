from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)

    # Role-based access control
    role = Column(String, default="engineer")  # admin, engineer, viewer, api_user

    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Tenant relationship
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)

    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")