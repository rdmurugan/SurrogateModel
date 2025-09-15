from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json


class SurrogateModelBase(ABC):
    """Abstract base class for all surrogate models"""

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters
        self.model = None
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        self.is_trained = False
        self.training_metrics = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying ML model with hyperparameters"""
        pass

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data"""
        pass

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        pass

    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty estimates for predictions"""
        pass

    def prepare_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale input data"""
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns)

        # Fit scalers on training data
        X_scaled = self.input_scaler.fit_transform(X.values)
        y_scaled = self.output_scaler.fit_transform(y.values)

        return X_scaled, y_scaled

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Train the surrogate model"""
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y)

        # Create and train model
        self.model = self._create_model()
        self._fit_model(X_scaled, y_scaled)

        # Calculate training metrics
        y_pred_scaled = self._predict_model(X_scaled)
        y_pred = self.output_scaler.inverse_transform(y_pred_scaled)

        self.training_metrics = self._calculate_metrics(y.values, y_pred)
        self.is_trained = True

        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with uncertainty quantification"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Scale input data
        X_scaled = self.input_scaler.transform(X.values)

        # Make predictions
        y_pred_scaled = self._predict_model(X_scaled)
        y_pred = self.output_scaler.inverse_transform(y_pred_scaled)

        # Get uncertainty estimates
        uncertainty = self.get_uncertainty(X_scaled)

        # Format output
        results = {}
        for i, target_name in enumerate(self.target_names):
            results[target_name] = {
                "prediction": float(y_pred[0, i]) if y_pred.ndim > 1 else float(y_pred[i]),
                "uncertainty": {k: float(v[i]) if hasattr(v[i], '__float__') else v[i]
                              for k, v in uncertainty.items()}
            }

        return results

    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame, cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        X_scaled, y_scaled = self.prepare_data(X, y)

        cv_scores = {}
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for i, target_name in enumerate(self.target_names):
            y_target = y_scaled[:, i] if y_scaled.ndim > 1 else y_scaled

            # Create temporary model for CV
            temp_model = self._create_model()

            # Perform cross-validation
            scores = cross_val_score(temp_model, X_scaled, y_target, cv=kfold, scoring='r2')
            cv_scores[f"{target_name}_r2_mean"] = float(np.mean(scores))
            cv_scores[f"{target_name}_r2_std"] = float(np.std(scores))

        return cv_scores

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}

        for i, target_name in enumerate(self.target_names):
            y_true_col = y_true[:, i] if y_true.ndim > 1 else y_true
            y_pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred

            metrics[f"{target_name}_r2"] = float(r2_score(y_true_col, y_pred_col))
            metrics[f"{target_name}_rmse"] = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
            metrics[f"{target_name}_mae"] = float(mean_absolute_error(y_true_col, y_pred_col))

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save the trained model and scalers"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'hyperparameters': self.hyperparameters,
            'training_metrics': self.training_metrics,
            'model_type': self.__class__.__name__
        }

        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model and scalers"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.input_scaler = model_data['input_scaler']
        self.output_scaler = model_data['output_scaler']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.hyperparameters = model_data['hyperparameters']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = True

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by the model"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        return None