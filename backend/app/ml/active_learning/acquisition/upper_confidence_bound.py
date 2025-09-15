import numpy as np
from typing import Tuple
from .base import AcquisitionFunction


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.

    UCB balances exploitation and exploration using confidence intervals.
    It's particularly effective when you want to bound the regret in optimization.

    Formula: UCB(x) = μ(x) + κ * σ(x)
    For minimization: LCB(x) = μ(x) - κ * σ(x)

    where κ (kappa) controls the exploration-exploitation trade-off.
    """

    def __init__(self, kappa: float = 2.0, minimize: bool = False, adaptive_kappa: bool = False, **kwargs):
        """
        Initialize Upper Confidence Bound acquisition function.

        Args:
            kappa: Confidence parameter. Higher values encourage exploration.
                   Typical range: 0.5 to 5.0
            minimize: Whether to minimize (True) or maximize (False) the objective
            adaptive_kappa: Whether to adapt kappa based on iteration count
        """
        super().__init__(**kwargs)
        self.kappa = kappa
        self.minimize = minimize
        self.adaptive_kappa = adaptive_kappa
        self.iteration = 0

    def evaluate(self, X: np.ndarray, model, **kwargs) -> np.ndarray:
        """
        Evaluate Upper Confidence Bound at given points.

        Args:
            X: Input points (n_points, n_features)
            model: Surrogate model with uncertainty estimation

        Returns:
            UCB values for each point (n_points,)
        """
        # Get current kappa value
        current_kappa = self._get_current_kappa(**kwargs)

        # Get model predictions and uncertainties
        mean_pred, std_pred = self._get_predictions(X, model)

        # Calculate UCB/LCB
        if self.minimize:
            # Lower Confidence Bound for minimization
            ucb_values = mean_pred - current_kappa * std_pred
        else:
            # Upper Confidence Bound for maximization
            ucb_values = mean_pred + current_kappa * std_pred

        return ucb_values

    def evaluate_gradients(self, X: np.ndarray, model, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate UCB and its gradients.

        For UCB, gradients are: ∇UCB = ∇μ(x) ± κ * ∇σ(x)
        """
        ucb_values = self.evaluate(X, model, **kwargs)

        # Finite difference gradients (most models don't provide analytical gradients)
        eps = 1e-6
        gradients = np.zeros_like(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += eps

            X_minus = X.copy()
            X_minus[:, i] -= eps

            ucb_plus = self.evaluate(X_plus, model, **kwargs)
            ucb_minus = self.evaluate(X_minus, model, **kwargs)

            gradients[:, i] = (ucb_plus - ucb_minus) / (2 * eps)

        return ucb_values, gradients

    def batch_evaluate(self, X: np.ndarray, model, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Evaluate UCB for batch active learning.

        UCB naturally supports batch selection by selecting points with highest
        UCB values, but we can enhance this with diversity considerations.
        """
        if batch_size == 1:
            return self.evaluate(X, model, **kwargs)

        # Get UCB values
        ucb_values = self.evaluate(X, model, **kwargs)

        # Add diversity penalty to encourage spatial diversity
        diversity_weight = kwargs.get('diversity_weight', 0.1)

        if diversity_weight > 0:
            ucb_values = self._add_diversity_penalty(X, ucb_values, batch_size, diversity_weight)

        return ucb_values

    def _add_diversity_penalty(self, X: np.ndarray, ucb_values: np.ndarray,
                              batch_size: int, diversity_weight: float) -> np.ndarray:
        """Add diversity penalty to encourage spatially diverse batch selection"""
        from scipy.spatial.distance import pdist, squareform

        # Calculate pairwise distances
        distances = squareform(pdist(X))

        # Greedy selection with diversity
        selected_indices = []
        remaining_indices = list(range(len(X)))
        adjusted_values = ucb_values.copy()

        for _ in range(min(batch_size, len(X))):
            if not remaining_indices:
                break

            # Select point with highest adjusted UCB
            best_idx_in_remaining = np.argmax(adjusted_values[remaining_indices])
            best_idx = remaining_indices[best_idx_in_remaining]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            # Apply diversity penalty to nearby points
            if remaining_indices:
                for idx in remaining_indices:
                    distance = distances[best_idx, idx]
                    # Penalty decreases with distance
                    penalty = diversity_weight * np.exp(-distance)
                    adjusted_values[idx] -= penalty

        # Create result array with higher values for selected points
        result = np.zeros_like(ucb_values)
        for i, idx in enumerate(selected_indices):
            result[idx] = ucb_values[idx] + (len(selected_indices) - i)

        return result

    def _get_current_kappa(self, **kwargs) -> float:
        """Get current kappa value, potentially adapted based on iteration"""
        if not self.adaptive_kappa:
            return self.kappa

        iteration = kwargs.get('iteration', self.iteration)
        total_iterations = kwargs.get('total_iterations', 100)

        # Adaptive kappa strategies
        strategy = kwargs.get('adaptive_strategy', 'decreasing')

        if strategy == 'decreasing':
            # Start high, decrease over time
            return self.kappa * np.sqrt(np.log(iteration + 1) / (iteration + 1))

        elif strategy == 'increasing':
            # Start low, increase over time (more exploration later)
            return self.kappa * np.sqrt(np.log(iteration + 1))

        elif strategy == 'cyclical':
            # Cyclical exploration/exploitation
            cycle_length = total_iterations // 4
            cycle_position = iteration % cycle_length
            return self.kappa * (1 + 0.5 * np.sin(2 * np.pi * cycle_position / cycle_length))

        else:
            return self.kappa

    def _get_predictions(self, X: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method to get predictions from various model types"""
        if hasattr(model, 'predict'):
            # For sklearn-like models
            mean_pred = model.predict(X)
            if hasattr(model, 'predict_std'):
                std_pred = model.predict_std(X)
            elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                # Gaussian Process case
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

    def set_iteration(self, iteration: int):
        """Set current iteration for adaptive kappa"""
        self.iteration = iteration

    def confidence_region_volume(self, X: np.ndarray, model, confidence_level: float = 0.95) -> float:
        """
        Calculate the volume of the confidence region.

        This can be used to assess the exploration progress.
        """
        mean_pred, std_pred = self._get_predictions(X, model)

        # Calculate confidence interval width
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        interval_widths = 2 * z_score * std_pred

        # Approximate volume as product of interval widths
        # (This is a simplification for high-dimensional spaces)
        volume = np.prod(np.mean(interval_widths))

        return volume

    @staticmethod
    def get_default_hyperparameters():
        """Get default hyperparameters for Upper Confidence Bound"""
        return {
            'kappa': [0.5, 1.0, 2.0, 3.0, 5.0],
            'minimize': [True, False],
            'adaptive_kappa': [True, False],
            'adaptive_strategy': ['decreasing', 'increasing', 'cyclical']
        }