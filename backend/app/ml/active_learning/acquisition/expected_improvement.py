import numpy as np
from scipy import stats
from typing import Tuple, Optional
from .base import AcquisitionFunction


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    EI balances exploration and exploitation by measuring the expected amount
    of improvement over the current best observation. Widely used in Bayesian
    optimization and proven effective for engineering optimization problems.

    Formula: EI(x) = E[max(f(x) - f_best, 0)]
    For Gaussian distribution: EI(x) = (μ(x) - f_best) * Φ(Z) + σ(x) * φ(Z)
    where Z = (μ(x) - f_best) / σ(x)
    """

    def __init__(self, xi: float = 0.01, minimize: bool = False, **kwargs):
        """
        Initialize Expected Improvement acquisition function.

        Args:
            xi: Exploration parameter (ξ). Higher values encourage exploration.
                Typical range: 0.001 to 0.1
            minimize: Whether to minimize (True) or maximize (False) the objective
        """
        super().__init__(**kwargs)
        self.xi = xi
        self.minimize = minimize

    def evaluate(self, X: np.ndarray, model, **kwargs) -> np.ndarray:
        """
        Evaluate Expected Improvement at given points.

        Args:
            X: Input points (n_points, n_features)
            model: Surrogate model with predict method returning mean and std

        Returns:
            EI values for each point (n_points,)
        """
        # Get model predictions
        if hasattr(model, 'predict'):
            # For sklearn-like models
            mean_pred = model.predict(X)
            if hasattr(model, 'predict_std'):
                std_pred = model.predict_std(X)
            elif hasattr(model, 'predict') and hasattr(model.model, 'predict'):
                # For our surrogate models with uncertainty
                if hasattr(model.model, 'predict'):
                    # Gaussian Process case
                    mean_pred, std_pred = model.model.predict(X, return_std=True)
                else:
                    # Fallback: assume 5% uncertainty
                    std_pred = np.abs(mean_pred) * 0.05
            else:
                std_pred = np.abs(mean_pred) * 0.05
        else:
            # For our custom surrogate models
            predictions = []
            uncertainties = []
            for i in range(len(X)):
                result = model.predict(X[i:i+1])
                # Extract first output for simplicity
                first_output = list(result.keys())[0]
                predictions.append(result[first_output]['prediction'])
                uncertainties.append(result[first_output]['uncertainty']['standard_deviation'])

            mean_pred = np.array(predictions)
            std_pred = np.array(uncertainties)

        # Ensure arrays are 1D
        if mean_pred.ndim > 1:
            mean_pred = mean_pred.flatten()
        if std_pred.ndim > 1:
            std_pred = std_pred.flatten()

        return self._calculate_ei(mean_pred, std_pred)

    def _calculate_ei(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate Expected Improvement given mean and standard deviation"""
        if self.current_best is None:
            raise ValueError("Current best value must be set before evaluation")

        # Handle the minimize/maximize case
        if self.minimize:
            improvement = self.current_best - mean - self.xi
        else:
            improvement = mean - self.current_best - self.xi

        # Avoid division by zero
        std = np.maximum(std, 1e-9)

        # Standardized improvement
        z = improvement / std

        # Expected Improvement formula
        ei = improvement * stats.norm.cdf(z) + std * stats.norm.pdf(z)

        # Handle numerical issues
        ei = np.maximum(ei, 0.0)

        return ei

    def evaluate_gradients(self, X: np.ndarray, model, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate EI and its gradients with respect to input parameters.

        This is useful for gradient-based optimization of the acquisition function.
        """
        # For most surrogate models, analytical gradients are not available
        # We use finite differences as an approximation
        ei_values = self.evaluate(X, model, **kwargs)

        # Finite difference gradients
        eps = 1e-6
        gradients = np.zeros_like(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += eps

            X_minus = X.copy()
            X_minus[:, i] -= eps

            ei_plus = self.evaluate(X_plus, model, **kwargs)
            ei_minus = self.evaluate(X_minus, model, **kwargs)

            gradients[:, i] = (ei_plus - ei_minus) / (2 * eps)

        return ei_values, gradients

    def batch_evaluate(self, X: np.ndarray, model, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Evaluate EI for batch active learning using q-Expected Improvement.

        This implements a more sophisticated batch strategy than the base class.
        """
        if batch_size == 1:
            return self.evaluate(X, model, **kwargs)

        # q-Expected Improvement (simplified Monte Carlo approximation)
        n_monte_carlo = kwargs.get('n_monte_carlo', 500)

        # Get model predictions
        mean_pred, std_pred = self._get_predictions(X, model)

        # Monte Carlo sampling for batch EI
        batch_ei = np.zeros(len(X))

        for mc_iter in range(n_monte_carlo):
            # Sample from posterior
            samples = np.random.normal(mean_pred, std_pred)

            # Find top batch_size samples
            if self.minimize:
                top_indices = np.argsort(samples)[:batch_size]
            else:
                top_indices = np.argsort(samples)[-batch_size:]

            # Calculate improvement for this MC sample
            if self.minimize:
                improvement = self.current_best - samples
            else:
                improvement = samples - self.current_best

            improvement = np.maximum(improvement - self.xi, 0)

            # Add to batch EI for selected points
            for idx in top_indices:
                batch_ei[idx] += improvement[idx]

        return batch_ei / n_monte_carlo

    def _get_predictions(self, X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method to get predictions from various model types"""
        if hasattr(model, 'predict'):
            mean_pred = model.predict(X)
            if hasattr(model, 'predict_std'):
                std_pred = model.predict_std(X)
            else:
                std_pred = np.abs(mean_pred) * 0.05
        else:
            # Custom surrogate models
            predictions = []
            uncertainties = []
            for i in range(len(X)):
                result = model.predict(X[i:i+1])
                first_output = list(result.keys())[0]
                predictions.append(result[first_output]['prediction'])
                uncertainties.append(result[first_output]['uncertainty']['standard_deviation'])

            mean_pred = np.array(predictions)
            std_pred = np.array(uncertainties)

        return mean_pred.flatten(), std_pred.flatten()

    def adaptive_xi(self, iteration: int, max_iterations: int) -> float:
        """
        Adaptively adjust exploration parameter xi during optimization.

        Args:
            iteration: Current iteration number
            max_iterations: Total number of iterations

        Returns:
            Adjusted xi value
        """
        # Start with high exploration, gradually reduce
        initial_xi = 0.1
        final_xi = 0.001

        # Exponential decay
        decay_rate = np.log(final_xi / initial_xi) / max_iterations
        adjusted_xi = initial_xi * np.exp(decay_rate * iteration)

        return max(adjusted_xi, final_xi)

    @staticmethod
    def get_default_hyperparameters():
        """Get default hyperparameters for Expected Improvement"""
        return {
            'xi': [0.001, 0.01, 0.05, 0.1],
            'minimize': [True, False]
        }