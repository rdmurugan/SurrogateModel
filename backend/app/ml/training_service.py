import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from sqlalchemy.orm import Session
from app.models.dataset import Dataset
from app.models.surrogate_model import SurrogateModel
from app.core.config import settings
from .factory import SurrogateModelFactory
from .utils.hyperparameter_optimizer import HyperparameterOptimizer
from .utils.model_validator import ModelValidator

logger = logging.getLogger(__name__)


class ModelTrainingService:
    """Service for training surrogate models"""

    def __init__(self, db: Session):
        self.db = db
        self.model_storage_path = Path("models")
        self.model_storage_path.mkdir(exist_ok=True)

    async def train_model(self, model_id: int) -> Dict[str, Any]:
        """
        Train a surrogate model asynchronously.

        Args:
            model_id: Database ID of the model to train

        Returns:
            Dictionary with training results and metrics
        """
        # Get model from database
        model_record = self.db.query(SurrogateModel).filter(
            SurrogateModel.id == model_id
        ).first()

        if not model_record:
            raise ValueError(f"Model with ID {model_id} not found")

        try:
            # Update status
            model_record.training_status = "training"
            model_record.training_start_time = datetime.utcnow()
            self.db.commit()

            # Load dataset
            dataset = self.db.query(Dataset).filter(
                Dataset.id == model_record.dataset_id
            ).first()

            if not dataset:
                raise ValueError(f"Dataset with ID {model_record.dataset_id} not found")

            # Load data
            X, y = await self._load_dataset(dataset)

            # Create and configure model
            surrogate_model = SurrogateModelFactory.create_model(
                algorithm=model_record.algorithm,
                **model_record.hyperparameters
            )

            # Train model
            training_metrics = surrogate_model.fit(X, y)

            # Validate model
            validator = ModelValidator()
            validation_metrics = validator.validate_model(surrogate_model, X, y)

            # Combine metrics
            all_metrics = {**training_metrics, **validation_metrics}

            # Save model to disk
            model_file_path = await self._save_model(surrogate_model, model_record)

            # Update database record
            model_record.training_status = "completed"
            model_record.training_end_time = datetime.utcnow()
            model_record.validation_metrics = all_metrics
            model_record.model_file_path = str(model_file_path)
            model_record.is_deployed = True

            self.db.commit()

            logger.info(f"Successfully trained model {model_id}")

            return {
                "model_id": model_id,
                "status": "completed",
                "metrics": all_metrics,
                "model_file_path": str(model_file_path)
            }

        except Exception as e:
            # Update status on failure
            model_record.training_status = "failed"
            model_record.training_end_time = datetime.utcnow()
            model_record.training_log = str(e)
            self.db.commit()

            logger.error(f"Failed to train model {model_id}: {str(e)}")
            raise

    async def train_with_hyperparameter_optimization(
        self,
        model_id: int,
        optimization_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a model with hyperparameter optimization.

        Args:
            model_id: Database ID of the model to train
            optimization_config: Configuration for hyperparameter optimization

        Returns:
            Dictionary with training results and best hyperparameters
        """
        # Get model from database
        model_record = self.db.query(SurrogateModel).filter(
            SurrogateModel.id == model_id
        ).first()

        if not model_record:
            raise ValueError(f"Model with ID {model_id} not found")

        try:
            # Update status
            model_record.training_status = "training"
            model_record.training_start_time = datetime.utcnow()
            self.db.commit()

            # Load dataset
            dataset = self.db.query(Dataset).filter(
                Dataset.id == model_record.dataset_id
            ).first()

            X, y = await self._load_dataset(dataset)

            # Configure optimizer
            optimizer = HyperparameterOptimizer()

            # Run optimization
            best_hyperparameters, optimization_results = await optimizer.optimize(
                algorithm=model_record.algorithm,
                X=X,
                y=y,
                config=optimization_config or {}
            )

            # Train final model with best hyperparameters
            surrogate_model = SurrogateModelFactory.create_model(
                algorithm=model_record.algorithm,
                **best_hyperparameters
            )

            training_metrics = surrogate_model.fit(X, y)

            # Validate model
            validator = ModelValidator()
            validation_metrics = validator.validate_model(surrogate_model, X, y)

            # Combine metrics
            all_metrics = {**training_metrics, **validation_metrics}

            # Save model to disk
            model_file_path = await self._save_model(surrogate_model, model_record)

            # Update database record
            model_record.hyperparameters = best_hyperparameters
            model_record.training_status = "completed"
            model_record.training_end_time = datetime.utcnow()
            model_record.validation_metrics = all_metrics
            model_record.model_file_path = str(model_file_path)
            model_record.is_deployed = True

            self.db.commit()

            return {
                "model_id": model_id,
                "status": "completed",
                "best_hyperparameters": best_hyperparameters,
                "optimization_results": optimization_results,
                "metrics": all_metrics,
                "model_file_path": str(model_file_path)
            }

        except Exception as e:
            model_record.training_status = "failed"
            model_record.training_end_time = datetime.utcnow()
            model_record.training_log = str(e)
            self.db.commit()

            logger.error(f"Failed to train model with HPO {model_id}: {str(e)}")
            raise

    async def _load_dataset(self, dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from storage"""
        # For now, simulate loading data
        # In production, this would load from actual file storage (MinIO/S3)

        # Generate synthetic data based on dataset metadata
        n_samples = dataset.num_samples
        input_columns = dataset.input_columns
        output_columns = dataset.output_columns

        # Create synthetic data for demonstration
        np.random.seed(42)

        X_data = {}
        for col in input_columns:
            if col in dataset.data_statistics:
                stats = dataset.data_statistics[col]
                X_data[col] = np.random.uniform(
                    stats['min'], stats['max'], n_samples
                )
            else:
                X_data[col] = np.random.randn(n_samples)

        y_data = {}
        for col in output_columns:
            if col in dataset.data_statistics:
                stats = dataset.data_statistics[col]
                y_data[col] = np.random.uniform(
                    stats['min'], stats['max'], n_samples
                )
            else:
                y_data[col] = np.random.randn(n_samples)

        X = pd.DataFrame(X_data)
        y = pd.DataFrame(y_data)

        return X, y

    async def _save_model(self, surrogate_model, model_record: SurrogateModel) -> Path:
        """Save trained model to disk"""
        # Create model-specific directory
        model_dir = self.model_storage_path / f"model_{model_record.id}"
        model_dir.mkdir(exist_ok=True)

        # Save model
        model_file_path = model_dir / "model.joblib"
        surrogate_model.save_model(str(model_file_path))

        # Save metadata
        metadata = {
            "model_id": model_record.id,
            "algorithm": model_record.algorithm,
            "hyperparameters": model_record.hyperparameters,
            "created_at": datetime.utcnow().isoformat(),
            "feature_names": surrogate_model.feature_names,
            "target_names": surrogate_model.target_names
        }

        metadata_file_path = model_dir / "metadata.json"
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_file_path

    def load_model(self, model_id: int):
        """Load a trained model from disk"""
        # Get model record
        model_record = self.db.query(SurrogateModel).filter(
            SurrogateModel.id == model_id
        ).first()

        if not model_record or not model_record.model_file_path:
            raise ValueError(f"Model {model_id} not found or not trained")

        # Create model instance
        surrogate_model = SurrogateModelFactory.create_model(
            algorithm=model_record.algorithm,
            **model_record.hyperparameters
        )

        # Load from disk
        surrogate_model.load_model(model_record.model_file_path)

        return surrogate_model

    async def predict(self, model_id: int, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        surrogate_model = self.load_model(model_id)

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        result = surrogate_model.predict(input_df)

        return result

    def get_model_info(self, model_id: int) -> Dict[str, Any]:
        """Get detailed information about a trained model"""
        model_record = self.db.query(SurrogateModel).filter(
            SurrogateModel.id == model_id
        ).first()

        if not model_record:
            raise ValueError(f"Model {model_id} not found")

        info = {
            "model_id": model_id,
            "algorithm": model_record.algorithm,
            "hyperparameters": model_record.hyperparameters,
            "training_status": model_record.training_status,
            "validation_metrics": model_record.validation_metrics,
            "is_deployed": model_record.is_deployed,
            "created_at": model_record.created_at.isoformat() if model_record.created_at else None,
            "training_duration": None
        }

        if model_record.training_start_time and model_record.training_end_time:
            duration = model_record.training_end_time - model_record.training_start_time
            info["training_duration"] = duration.total_seconds()

        # Add algorithm-specific info if model is trained
        if model_record.is_deployed and model_record.model_file_path:
            try:
                surrogate_model = self.load_model(model_id)

                # Get feature importance if available
                feature_importance = surrogate_model.get_feature_importance()
                if feature_importance:
                    info["feature_importance"] = feature_importance

                # Get algorithm-specific information
                if hasattr(surrogate_model, 'get_sobol_indices'):
                    info["sobol_indices"] = surrogate_model.get_sobol_indices()

                if hasattr(surrogate_model, 'get_oob_score'):
                    info["oob_score"] = surrogate_model.get_oob_score()

                if hasattr(surrogate_model, 'get_training_history'):
                    info["training_history"] = surrogate_model.get_training_history()

            except Exception as e:
                logger.warning(f"Could not load additional model info for {model_id}: {e}")

        return info