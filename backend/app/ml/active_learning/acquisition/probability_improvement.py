import numpy as np
from scipy import stats
from typing import Tuple
from .base import AcquisitionFunction


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.

    PI calculates the probability that a point will improve over the current best.
    It's simpler than Expected Improvement but can be overly exploitative,
    focusing too much on areas near the current best.

    Formula: PI(x) = P(f(x) > f_best + ξ)
    For Gaussian: PI(x) = Φ((μ(x) - f_best - ξ) / σ(x))
    """

    def __init__(self, xi: float = 0.01, minimize: bool = False, **kwargs):
        """
        Initialize Probability of Improvement acquisition function.

        Args:
            xi: Exploration parameter (ξ). Small positive value to ensure improvement.
                Typical range: 0.001 to 0.1
            minimize: Whether to minimize (True) or maximize (False) the objective
        """
        super().__init__(**kwargs)
        self.xi = xi
        self.minimize = minimize

    def evaluate(self, X: np.ndarray, model, **kwargs) -> np.ndarray:
        """
        Evaluate Probability of Improvement at given points.

        Args:
            X: Input points (n_points, n_features)
            model: Surrogate model with uncertainty estimation

        Returns:
            PI values for each point (n_points,)
        """
        # Get model predictions and uncertainties
        mean_pred, std_pred = self._get_predictions(X, model)

        return self._calculate_pi(mean_pred, std_pred)

    def _calculate_pi(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate Probability of Improvement given mean and standard deviation"""
        if self.current_best is None:
            raise ValueError("Current best value must be set before evaluation")

        # Handle the minimize/maximize case
        if self.minimize:
            # For minimization: P(f(x) < f_best - ξ)
            improvement_threshold = self.current_best - self.xi
            z = (improvement_threshold - mean) / np.maximum(std, 1e-9)
        else:
            # For maximization: P(f(x) > f_best + ξ)
            improvement_threshold = self.current_best + self.xi
            z = (mean - improvement_threshold) / np.maximum(std, 1e-9)

        # Probability of improvement using cumulative distribution function
        pi = stats.norm.cdf(z)

        # Handle numerical issues
        pi = np.clip(pi, 0.0, 1.0)

        return pi

    def evaluate_gradients(self, X: np.ndarray, model, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate PI and its gradients.

        For PI: ∇PI = φ(z) * (∇μ/σ - μ*∇σ/σ²) where z = (μ - f_best - ξ)/σ
        """
        pi_values = self.evaluate(X, model, **kwargs)

        # Use finite differences for gradient approximation
        eps = 1e-6
        gradients = np.zeros_like(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += eps

            X_minus = X.copy()
            X_minus[:, i] -= eps

            pi_plus = self.evaluate(X_plus, model, **kwargs)
            pi_minus = self.evaluate(X_minus, model, **kwargs)

            gradients[:, i] = (pi_plus - pi_minus) / (2 * eps)

        return pi_values, gradients

    def batch_evaluate(self, X: np.ndarray, model, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Evaluate PI for batch active learning.

        Since PI is prone to being overly exploitative, we add diversity
        considerations for batch selection.
        """
        if batch_size == 1:
            return self.evaluate(X, model, **kwargs)

        # Get individual PI values
        pi_values = self.evaluate(X, model, **kwargs)

        # Add diversity penalty similar to UCB
        diversity_weight = kwargs.get('diversity_weight', 0.2)  # Higher than UCB default

        if diversity_weight > 0:
            pi_values = self._add_diversity_penalty(X, pi_values, batch_size, diversity_weight)

        return pi_values

    def _add_diversity_penalty(self, X: np.ndarray, pi_values: np.ndarray,
                              batch_size: int, diversity_weight: float) -> np.ndarray:
        """Add diversity penalty to encourage spatially diverse batch selection"""
        from scipy.spatial.distance import pdist, squareform

        # Calculate pairwise distances
        distances = squareform(pdist(X))

        # Greedy selection with diversity
        selected_indices = []
        remaining_indices = list(range(len(X)))
        adjusted_values = pi_values.copy()

        for _ in range(min(batch_size, len(X))):
            if not remaining_indices:
                break

            # Select point with highest adjusted PI
            best_idx_in_remaining = np.argmax(adjusted_values[remaining_indices])
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            # Apply diversity penalty to nearby points
            if remaining_indices:
                for idx in remaining_indices:
                    distance = distances[best_idx, idx]
                    # Penalty decreases with distance
                    penalty = diversity_weight * pi_values[best_idx] * np.exp(-distance)
                    adjusted_values[idx] -= penalty

        # Create result array with higher values for selected points
        result = np.zeros_like(pi_values)
        for i, idx in enumerate(selected_indices):
            result[idx] = pi_values[idx] + (len(selected_indices) - i) * 0.1

        return result

    def _get_predictions(self, X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method to get predictions from various model types"""
        if hasattr(model, 'predict'):
            mean_pred = model.predict(X)
            if hasattr(model, 'predict_std'):
                std_pred = model.predict_std(X)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                try:
                    mean_pred, std_pred = model.model.predict(X, return_std=True)
                except:
                    std_pred = np.abs(mean_pred) * 0.05
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

    def improvement_probability_contours(self, X_grid: np.ndarray, model,
                                       probability_levels: list = [0.1, 0.5, 0.9]) -> dict:
        """
        Calculate probability of improvement contours for visualization.

        Args:
            X_grid: Grid of points for contour calculation
            model: Trained model
            probability_levels: Probability levels for contours

        Returns:
            Dictionary with contour data
        """
        pi_values = self.evaluate(X_grid, model)

        contours = {}
        for level in probability_levels:
            # Find points where PI >= level
            mask = pi_values >= level
            contours[f'pi_{level}'] = {
                'points': X_grid[mask],
                'values': pi_values[mask]
            }

        return contours

    def adaptive_xi_strategy(self, iteration: int, max_iterations: int,
                           strategy: str = 'decreasing') -> float:
        """
        Adaptively adjust xi parameter during optimization.

        Args:
            iteration: Current iteration
            max_iterations: Total iterations
            strategy: Adaptation strategy ('decreasing', 'increasing', 'cyclic')

        Returns:
            Adjusted xi value
        """
        if strategy == 'decreasing':
            # Start with exploration, become more exploitative
            return self.xi * (1 - iteration / max_iterations)

        elif strategy == 'increasing':
            # Start exploitative, become more exploratory
            return self.xi * (iteration / max_iterations)

        elif strategy == 'cyclic':
            # Alternate between exploration and exploitation
            cycle_position = (iteration % (max_iterations // 4)) / (max_iterations // 4)
            return self.xi * (1 + np.sin(2 * np.pi * cycle_position))

        else:
            return self.xi

    def risk_adjusted_pi(self, X: np.ndarray, model, risk_aversion: float = 1.0) -> np.ndarray:
        """
        Calculate risk-adjusted probability of improvement.

        This variant considers the uncertainty in the improvement prediction itself.

        Args:
            X: Input points
            model: Trained model
            risk_aversion: Risk aversion parameter (0 = risk-neutral, >0 = risk-averse)

        Returns:
            Risk-adjusted PI values
        """
        mean_pred, std_pred = self._get_predictions(X, model)

        # Standard PI
        pi_values = self._calculate_pi(mean_pred, std_pred)

        # Risk adjustment based on prediction uncertainty
        risk_penalty = risk_aversion * std_pred / (np.mean(std_pred) + 1e-9)
        risk_adjusted_pi = pi_values * np.exp(-risk_penalty)

        return risk_adjusted_pi

    @staticmethod
    def get_default_hyperparameters():
        """Get default hyperparameters for Probability of Improvement"""
        return {
            'xi': [0.001, 0.01, 0.05, 0.1],
            'minimize': [True, False]
        }