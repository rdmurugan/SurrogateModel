from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)

    # File information
    file_path = Column(String, nullable=False)  # Path in object storage
    file_size_mb = Column(Float)
    file_format = Column(String)  # csv, hdf5, etc.

    # Data structure
    input_columns = Column(JSON)  # List of input parameter names
    output_columns = Column(JSON)  # List of output parameter names
    num_samples = Column(Integer)

    # Data statistics
    data_statistics = Column(JSON)  # Min, max, mean, std for each column

    # Processing status
    status = Column(String, default="uploaded")  # uploaded, processing, processed, error
    processing_log = Column(String)

    # Tenant relationship
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)

    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="datasets")
    models = relationship("SurrogateModel", back_populates="dataset")