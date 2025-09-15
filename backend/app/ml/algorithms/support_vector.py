import numpy as np
from typing import Dict, Any
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from ..base import SurrogateModelBase


class SupportVectorSurrogate(SurrogateModelBase):
    """
    Support Vector Regression (SVR) surrogate model.

    Excellent for:
    - Medium-sized datasets
    - High-dimensional input spaces
    - Robust regression with outliers
    - Non-linear relationships with kernel trick

    Advantages:
    - Effective in high dimensions
    - Memory efficient (uses subset of training points)
    - Versatile with different kernel functions
    - Robust to outliers

    Disadvantages:
    - Sensitive to hyperparameters
    - No direct uncertainty quantification
    - Slow training on large datasets
    - Limited interpretability
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale',
            'degree': 3,  # For polynomial kernel
            'coef0': 0.0,  # For polynomial and sigmoid kernels
            'shrinking': True,
            'cache_size': 200,
            'max_iter': -1
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)

    def _create_model(self) -> SVR:
        """Create Support Vector Regression model"""
        svr = SVR(
            kernel=self.hyperparameters.get('kernel', 'rbf'),
            C=self.hyperparameters.get('C', 1.0),
            epsilon=self.hyperparameters.get('epsilon', 0.1),
            gamma=self.hyperparameters.get('gamma', 'scale'),
            degree=self.hyperparameters.get('degree', 3),
            coef0=self.hyperparameters.get('coef0', 0.0),
            shrinking=self.hyperparameters.get('shrinking', True),
            cache_size=self.hyperparameters.get('cache_size', 200),
            max_iter=self.hyperparameters.get('max_iter', -1)
        )

        # For multi-output problems, wrap with MultiOutputRegressor
        if len(self.target_names) > 1:
            return MultiOutputRegressor(svr)
        else:
            return svr

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the SVR model"""
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with SVR"""
        predictions = self.model.predict(X)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates for SVR.

        SVR doesn't provide direct uncertainty estimates, so we use
        different approaches based on the model structure.
        """
        predictions = self._predict_model(X)

        uncertainty = {}
        for i, target_name in enumerate(self.target_names):
            pred_val = float(predictions[0, i]) if predictions.ndim > 1 else float(predictions[0])

            # Method 1: Distance to support vectors (for RBF kernel)
            if hasattr(self.model, 'support_vectors_') and self.hyperparameters.get('kernel') == 'rbf':
                uncertainty_estimate = self._distance_based_uncertainty(X, target_name)
            else:
                # Method 2: Bootstrap or cross-validation based uncertainty
                uncertainty_estimate = self._bootstrap_uncertainty(X, target_name)

            uncertainty[target_name] = uncertainty_estimate

        return uncertainty

    def _distance_based_uncertainty(self, X: np.ndarray, target_name: str) -> Dict[str, float]:
        """
        Estimate uncertainty based on distance to support vectors.

        Points far from support vectors are more uncertain.
        """
        if len(self.target_names) > 1:
            # For multi-output, get the specific estimator
            estimator_idx = self.target_names.index(target_name)
            estimator = self.model.estimators_[estimator_idx]
        else:
            estimator = self.model

        if not hasattr(estimator, 'support_vectors_'):
            return self._default_uncertainty(X, target_name)

        # Calculate minimum distance to support vectors
        support_vectors = estimator.support_vectors_
        distances = []

        for sv in support_vectors:
            dist = np.linalg.norm(X[0] - sv)
            distances.append(dist)

        min_distance = min(distances)
        avg_distance = np.mean(distances)

        # Convert distance to uncertainty estimate
        # Higher distance = higher uncertainty
        gamma = self.hyperparameters.get('gamma', 'scale')
        if gamma == 'scale':
            gamma = 1.0 / X.shape[1]
        elif gamma == 'auto':
            gamma = 1.0 / X.shape[1]

        # Exponential decay of confidence with distance
        confidence = np.exp(-gamma * min_distance)
        uncertainty_factor = 1.0 - confidence

        pred_val = float(self._predict_model(X)[0, self.target_names.index(target_name)])
        std_estimate = abs(pred_val) * uncertainty_factor * 0.1  # Scale factor

        return {
            'standard_deviation': std_estimate,
            'variance': std_estimate**2,
            'confidence_interval_95': [
                pred_val - 1.96 * std_estimate,
                pred_val + 1.96 * std_estimate
            ],
            'distance_to_support': float(min_distance),
            'confidence_score': float(confidence)
        }

    def _bootstrap_uncertainty(self, X: np.ndarray, target_name: str) -> Dict[str, float]:
        """
        Estimate uncertainty using bootstrap sampling.

        This requires retraining models which is computationally expensive,
        so we use a simplified approach based on model complexity.
        """
        pred_val = float(self._predict_model(X)[0, self.target_names.index(target_name)])

        # Estimate uncertainty based on model complexity and data size
        n_support_vectors = self._get_n_support_vectors(target_name)
        data_size = len(self.input_scaler.mean_) if hasattr(self.input_scaler, 'mean_') else 100

        # Higher support vector ratio suggests more complex decision boundary
        complexity_factor = n_support_vectors / data_size if data_size > 0 else 0.5
        std_estimate = abs(pred_val) * complexity_factor * 0.05

        return {
            'standard_deviation': std_estimate,
            'variance': std_estimate**2,
            'confidence_interval_95': [
                pred_val - 1.96 * std_estimate,
                pred_val + 1.96 * std_estimate
            ],
            'n_support_vectors': n_support_vectors,
            'complexity_factor': float(complexity_factor)
        }

    def _default_uncertainty(self, X: np.ndarray, target_name: str) -> Dict[str, float]:
        """Default uncertainty estimate when other methods are not available"""
        pred_val = float(self._predict_model(X)[0, self.target_names.index(target_name)])
        std_estimate = abs(pred_val) * 0.05  # 5% of prediction

        return {
            'standard_deviation': std_estimate,
            'variance': std_estimate**2,
            'confidence_interval_95': [
                pred_val - 1.96 * std_estimate,
                pred_val + 1.96 * std_estimate
            ]
        }

    def _get_n_support_vectors(self, target_name: str) -> int:
        """Get number of support vectors for a specific output"""
        if len(self.target_names) > 1:
            estimator_idx = self.target_names.index(target_name)
            estimator = self.model.estimators_[estimator_idx]
        else:
            estimator = self.model

        if hasattr(estimator, 'n_support_'):
            return int(np.sum(estimator.n_support_))
        elif hasattr(estimator, 'support_vectors_'):
            return len(estimator.support_vectors_)
        else:
            return 0

    def get_support_vector_info(self) -> Dict[str, Any]:
        """Get information about support vectors"""
        info = {}

        for i, target_name in enumerate(self.target_names):
            if len(self.target_names) > 1:
                estimator = self.model.estimators_[i]
            else:
                estimator = self.model

            target_info = {
                'n_support_vectors': self._get_n_support_vectors(target_name),
                'support_vector_ratio': 0.0
            }

            if hasattr(estimator, 'support_vectors_'):
                n_total = len(estimator.support_vectors_) + len(getattr(estimator, 'support_', []))
                if n_total > 0:
                    target_info['support_vector_ratio'] = target_info['n_support_vectors'] / n_total

            info[target_name] = target_info

        return info

    def get_kernel_parameters(self) -> Dict[str, Any]:
        """Get kernel parameters for analysis"""
        params = {}

        # Get parameters from first estimator
        estimator = self.model.estimators_[0] if len(self.target_names) > 1 else self.model

        if hasattr(estimator, 'gamma'):
            params['gamma'] = float(estimator.gamma) if isinstance(estimator.gamma, (int, float)) else estimator.gamma

        if hasattr(estimator, 'C'):
            params['C'] = float(estimator.C)

        if hasattr(estimator, 'epsilon'):
            params['epsilon'] = float(estimator.epsilon)

        params['kernel'] = self.hyperparameters.get('kernel', 'rbf')

        return params

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'degree': [2, 3, 4, 5]  # For polynomial kernel
        }