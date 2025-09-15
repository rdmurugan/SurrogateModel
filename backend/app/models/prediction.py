from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    # Input and output data
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)

    # Uncertainty quantification
    uncertainty_data = Column(JSON)  # Confidence intervals, std dev, etc.

    # Prediction metadata
    prediction_time_ms = Column(Float)
    batch_size = Column(Integer, default=1)

    # Model relationship
    model_id = Column(Integer, ForeignKey("surrogate_models.id"), nullable=False)

    # Metadata
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    model = relationship("SurrogateModel", back_populates="predictions")