from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    subdomain = Column(String, unique=True, nullable=False, index=True)

    # Subscription details
    subscription_tier = Column(String, default="starter")  # starter, professional, enterprise
    is_active = Column(Boolean, default=True)

    # Resource limits
    max_users = Column(Integer, default=1)
    max_models = Column(Integer, default=2)
    max_predictions_per_month = Column(Integer, default=1000)
    max_storage_gb = Column(Integer, default=5)

    # Metadata
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="tenant")
    datasets = relationship("Dataset", back_populates="tenant")
    models = relationship("SurrogateModel", back_populates="tenant")