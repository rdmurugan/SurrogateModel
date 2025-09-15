from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Surrogate Model Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5432/surrogate_platform"

    # Redis settings
    redis_url: str = "redis://localhost:6379"

    # Authentication settings
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # File storage settings
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False

    # ML settings
    max_training_time_minutes: int = 60
    max_file_size_mb: int = 100

    # CORS settings
    backend_cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    class Config:
        env_file = ".env"


settings = Settings()