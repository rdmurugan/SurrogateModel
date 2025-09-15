import numpy as np
from typing import Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from ..base import SurrogateModelBase


class GaussianProcessSurrogate(SurrogateModelBase):
    """
    Gaussian Process Regression surrogate model.

    Excellent for:
    - Small to medium datasets (< 10,000 samples)
    - Smooth functions
    - Uncertainty quantification
    - Global optimization

    Advantages:
    - Provides uncertainty estimates
    - Non-parametric
    - Good extrapolation capabilities
    - Handles noise well

    Disadvantages:
    - Computationally expensive for large datasets
    - Sensitive to hyperparameters
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'kernel_type': 'rbf',
            'length_scale': 1.0,
            'length_scale_bounds': (1e-5, 1e5),
            'nu': 1.5,  # For Matern kernel
            'alpha': 1e-10,
            'n_restarts_optimizer': 10,
            'normalize_y': True
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)

    def _create_model(self) -> GaussianProcessRegressor:
        """Create Gaussian Process model with specified kernel"""

        # Select kernel based on hyperparameters
        kernel_type = self.hyperparameters.get('kernel_type', 'rbf')
        length_scale = self.hyperparameters.get('length_scale', 1.0)
        length_scale_bounds = self.hyperparameters.get('length_scale_bounds', (1e-5, 1e5))

        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            )
        elif kernel_type == 'matern':
            nu = self.hyperparameters.get('nu', 1.5)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=nu
            )
        elif kernel_type == 'rbf_white':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds
            ) + WhiteKernel(1e-3, (1e-10, 1e+1))
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.hyperparameters.get('alpha', 1e-10),
            n_restarts_optimizer=self.hyperparameters.get('n_restarts_optimizer', 10),
            normalize_y=self.hyperparameters.get('normalize_y', True),
            random_state=42
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Gaussian Process model"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the GP model"""
        return self.model.predict(X)

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty estimates from GP"""
        mean_pred, std_pred = self.model.predict(X, return_std=True)

        # Calculate confidence intervals (95%)
        confidence_interval_95 = np.array([
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred
        ]).T

        uncertainty = {}
        for i, target_name in enumerate(self.target_names):
            if len(self.target_names) == 1:
                uncertainty[target_name] = {
                    'standard_deviation': float(std_pred[0]),
                    'variance': float(std_pred[0] ** 2),
                    'confidence_interval_95': confidence_interval_95[0].tolist()
                }
            else:
                uncertainty[target_name] = {
                    'standard_deviation': float(std_pred[0, i]),
                    'variance': float(std_pred[0, i] ** 2),
                    'confidence_interval_95': confidence_interval_95[0, i].tolist()
                }

        return uncertainty

    def acquisition_function(self, X: np.ndarray, acquisition_type: str = 'ei') -> np.ndarray:
        """
        Calculate acquisition function for active learning/optimization

        Args:
            X: Input points to evaluate
            acquisition_type: 'ei' (expected improvement), 'pi' (probability improvement), 'ucb' (upper confidence bound)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating acquisition function")

        mean, std = self.model.predict(X, return_std=True)

        if acquisition_type == 'ei':
            # Expected Improvement
            if hasattr(self, 'best_y'):
                best_y = self.best_y
            else:
                best_y = np.max(self.model.y_train_)

            z = (mean - best_y) / (std + 1e-9)
            ei = (mean - best_y) * self._norm_cdf(z) + std * self._norm_pdf(z)
            return ei

        elif acquisition_type == 'pi':
            # Probability of Improvement
            if hasattr(self, 'best_y'):
                best_y = self.best_y
            else:
                best_y = np.max(self.model.y_train_)

            z = (mean - best_y) / (std + 1e-9)
            return self._norm_cdf(z)

        elif acquisition_type == 'ucb':
            # Upper Confidence Bound
            kappa = self.hyperparameters.get('ucb_kappa', 2.0)
            return mean + kappa * std

        else:
            raise ValueError(f"Unknown acquisition type: {acquisition_type}")

    def _norm_cdf(self, x):
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _norm_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'kernel_type': ['rbf', 'matern', 'rbf_white'],
            'length_scale': [0.1, 1.0, 10.0],
            'nu': [0.5, 1.5, 2.5],  # For Matern kernel
            'alpha': [1e-10, 1e-8, 1e-6],
            'n_restarts_optimizer': [5, 10, 20]
        }