from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class SurrogateModel(Base):
    __tablename__ = "surrogate_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)

    # Model configuration
    algorithm = Column(String, nullable=False)  # gaussian_process, neural_network, polynomial_chaos, etc.
    hyperparameters = Column(JSON)

    # Model files
    model_file_path = Column(String)  # Path to serialized model
    scaler_file_path = Column(String)  # Path to input/output scalers

    # Training information
    training_status = Column(String, default="pending")  # pending, training, completed, failed
    training_start_time = Column(DateTime)
    training_end_time = Column(DateTime)
    training_log = Column(String)

    # Performance metrics
    validation_metrics = Column(JSON)  # R2, RMSE, etc.
    cross_validation_scores = Column(JSON)

    # Deployment
    is_deployed = Column(Boolean, default=False)
    endpoint_url = Column(String)

    # Relationships
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)

    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    dataset = relationship("Dataset", back_populates="models")
    tenant = relationship("Tenant", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")