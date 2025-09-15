from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class AcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions used in active learning.

    Acquisition functions guide the selection of new sample points by balancing
    exploration (uncertainty) and exploitation (promising regions).
    """

    def __init__(self, **kwargs):
        self.hyperparameters = kwargs
        self.current_best = None

    @abstractmethod
    def evaluate(self, X: np.ndarray, model, **kwargs) -> np.ndarray:
        """
        Evaluate the acquisition function at given points.

        Args:
            X: Input points to evaluate (n_points, n_features)
            model: Trained surrogate model with uncertainty estimation
            **kwargs: Additional parameters

        Returns:
            Acquisition values for each point (n_points,)
        """
        pass

    @abstractmethod
    def evaluate_gradients(self, X: np.ndarray, model, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate acquisition function and its gradients.

        Args:
            X: Input points to evaluate
            model: Trained surrogate model
            **kwargs: Additional parameters

        Returns:
            Tuple of (acquisition_values, gradients)
        """
        pass

    def set_current_best(self, best_value: float):
        """Set the current best observed value"""
        self.current_best = best_value

    def batch_evaluate(self, X: np.ndarray, model, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Evaluate acquisition function for batch active learning.

        This base implementation uses a greedy approach. Subclasses can override
        for more sophisticated batch strategies.
        """
        if batch_size == 1:
            return self.evaluate(X, model, **kwargs)

        # Greedy batch selection
        selected_indices = []
        remaining_indices = list(range(len(X)))

        for _ in range(min(batch_size, len(X))):
            if not remaining_indices:
                break

            # Evaluate remaining points
            X_remaining = X[remaining_indices]
            acq_values = self.evaluate(X_remaining, model, **kwargs)

            # Select best point
            best_idx = np.argmax(acq_values)
            selected_indices.append(remaining_indices[best_idx])
            remaining_indices.pop(best_idx)

            # Update model with selected point (simplified)
            # In practice, you might want to update the model or use fantasy points

        # Return acquisition values for selected points
        result = np.zeros(len(X))
        for i, idx in enumerate(selected_indices):
            result[idx] = len(selected_indices) - i  # Rank-based scoring

        return result

    def optimize_acquisition(self, model, bounds: np.ndarray, n_restarts: int = 10) -> Tuple[np.ndarray, float]:
        """
        Optimize the acquisition function to find the best next point.

        Args:
            model: Trained surrogate model
            bounds: Parameter bounds (n_features, 2)
            n_restarts: Number of optimization restarts

        Returns:
            Tuple of (best_point, best_acquisition_value)
        """
        from scipy.optimize import minimize

        best_x = None
        best_acq = -np.inf

        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])

            # Objective function (negative acquisition for minimization)
            def objective(x):
                x_2d = x.reshape(1, -1)
                acq_val = self.evaluate(x_2d, model)
                return -acq_val[0]

            # Gradient function if available
            def jac(x):
                x_2d = x.reshape(1, -1)
                _, grad = self.evaluate_gradients(x_2d, model)
                return -grad[0]

            # Bounds for optimization
            opt_bounds = [(bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]

            try:
                # Use gradients if available
                if hasattr(self, 'evaluate_gradients'):
                    result = minimize(objective, x0, method='L-BFGS-B',
                                    jac=jac, bounds=opt_bounds)
                else:
                    result = minimize(objective, x0, method='L-BFGS-B',
                                    bounds=opt_bounds)

                if result.success and -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x

            except Exception:
                # Fallback to simple evaluation
                acq_val = -objective(x0)
                if acq_val > best_acq:
                    best_acq = acq_val
                    best_x = x0

        return best_x, best_acq

    def _normalize_inputs(self, X: np.ndarray, bounds: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize inputs to [0, 1] range if bounds are provided"""
        if bounds is None:
            return X

        X_norm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        return X_norm

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters"""
        return self.hyperparameters.copy()

    def set_hyperparameters(self, **kwargs):
        """Update hyperparameters"""
        self.hyperparameters.update(kwargs)