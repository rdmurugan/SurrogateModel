import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any
from .base import AcquisitionFunction


class KnowledgeGradient(AcquisitionFunction):
    """
    Knowledge Gradient (KG) acquisition function.

    KG measures the expected value of information gained by sampling a point.
    It's particularly effective for problems where the goal is to quickly
    identify the global optimum rather than just improving the current best.

    KG considers how much the posterior belief about the optimum will improve
    after observing a new point, making it excellent for finite-horizon optimization.
    """

    def __init__(self, n_fantasy_points: int = 100, minimize: bool = False,
                 discrete_optimization: bool = False, **kwargs):
        """
        Initialize Knowledge Gradient acquisition function.

        Args:
            n_fantasy_points: Number of fantasy points for Monte Carlo estimation
            minimize: Whether to minimize (True) or maximize (False) the objective
            discrete_optimization: Whether the optimization is over discrete points
        """
        super().__init__(**kwargs)
        self.n_fantasy_points = n_fantasy_points
        self.minimize = minimize
        self.discrete_optimization = discrete_optimization

    def evaluate(self, X: np.ndarray, model, **kwargs) -> np.ndarray:
        """
        Evaluate Knowledge Gradient at given points.

        Args:
            X: Input points (n_points, n_features)
            model: Surrogate model with uncertainty estimation

        Returns:
            KG values for each point (n_points,)
        """
        # Get candidate points for optimization (if not provided)
        X_candidates = kwargs.get('X_candidates', X)

        kg_values = np.zeros(len(X))

        for i, x_new in enumerate(X):
            kg_values[i] = self._calculate_kg_single_point(x_new, X_candidates, model)

        return kg_values

    def _calculate_kg_single_point(self, x_new: np.ndarray, X_candidates: np.ndarray, model) -> float:
        """
        Calculate Knowledge Gradient for a single point.

        This implements the one-step look-ahead KG using Monte Carlo sampling.
        """
        # Get current posterior at the new point
        x_new_2d = x_new.reshape(1, -1)
        mean_new, std_new = self._get_predictions(x_new_2d, model)

        if std_new[0] < 1e-10:
            return 0.0  # No information gain if no uncertainty

        # Current best estimate
        mean_candidates, _ = self._get_predictions(X_candidates, model)

        if self.minimize:
            current_best_estimate = np.min(mean_candidates)
        else:
            current_best_estimate = np.max(mean_candidates)

        # Monte Carlo estimation of expected improvement in best value
        fantasy_values = np.random.normal(mean_new[0], std_new[0], self.n_fantasy_points)

        expected_improvements = []

        for fantasy_value in fantasy_values:
            # Update belief with fantasy observation
            updated_best = self._get_updated_best_estimate(
                x_new_2d, fantasy_value, X_candidates, model
            )

            # Calculate improvement
            if self.minimize:
                improvement = current_best_estimate - updated_best
            else:
                improvement = updated_best - current_best_estimate

            expected_improvements.append(max(improvement, 0))

        return np.mean(expected_improvements)

    def _get_updated_best_estimate(self, x_new: np.ndarray, y_new: float,
                                  X_candidates: np.ndarray, model) -> float:
        """
        Estimate the best value after hypothetically observing y_new at x_new.

        This is a simplified approximation. In practice, you might want to
        update the GP posterior exactly.
        """
        # Simple approximation: assume the fantasy point is the best if it's better
        mean_candidates, _ = self._get_predictions(X_candidates, model)

        all_means = np.append(mean_candidates, y_new)

        if self.minimize:
            return np.min(all_means)
        else:
            return np.max(all_means)

    def evaluate_gradients(self, X: np.ndarray, model, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate KG and its gradients.

        Note: KG gradients are complex to compute analytically.
        This uses finite differences.
        """
        kg_values = self.evaluate(X, model, **kwargs)

        # Finite difference gradients
        eps = 1e-6
        gradients = np.zeros_like(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += eps

            X_minus = X.copy()
            X_minus[:, i] -= eps

            kg_plus = self.evaluate(X_plus, model, **kwargs)
            kg_minus = self.evaluate(X_minus, model, **kwargs)

            gradients[:, i] = (kg_plus - kg_minus) / (2 * eps)

        return kg_values, gradients

    def batch_evaluate(self, X: np.ndarray, model, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Evaluate KG for batch active learning.

        This implements a sequential version of batch KG, where each point
        in the batch is selected considering the previous selections.
        """
        if batch_size == 1:
            return self.evaluate(X, model, **kwargs)

        # Sequential batch selection
        selected_indices = []
        remaining_indices = list(range(len(X)))
        kg_values = np.zeros(len(X))

        X_candidates = kwargs.get('X_candidates', X)

        for batch_idx in range(min(batch_size, len(X))):
            if not remaining_indices:
                break

            # Evaluate KG for remaining points
            temp_kg = np.zeros(len(remaining_indices))

            for i, idx in enumerate(remaining_indices):
                temp_kg[i] = self._calculate_kg_single_point(X[idx], X_candidates, model)

            # Select best point
            best_local_idx = np.argmax(temp_kg)
            best_global_idx = remaining_indices[best_local_idx]

            selected_indices.append(best_global_idx)
            kg_values[best_global_idx] = temp_kg[best_local_idx] + (batch_size - batch_idx)

            remaining_indices.remove(best_global_idx)

            # TODO: Update model with fantasy point for next iteration
            # This would require more sophisticated posterior updating

        return kg_values

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

    def value_of_information(self, X: np.ndarray, model) -> Dict[str, float]:
        """
        Calculate various information-theoretic metrics.

        Returns:
            Dictionary with VoI metrics
        """
        mean_pred, std_pred = self._get_predictions(X, model)

        # Entropy of current posterior
        entropy = 0.5 * np.log(2 * np.pi * np.e * std_pred**2)
        total_entropy = np.sum(entropy)

        # Expected entropy reduction (simplified)
        expected_entropy_reduction = np.mean(entropy)

        # Mutual information approximation
        mutual_info = np.mean(np.log(std_pred + 1e-10))

        return {
            'total_entropy': float(total_entropy),
            'mean_entropy': float(np.mean(entropy)),
            'expected_entropy_reduction': float(expected_entropy_reduction),
            'mutual_information': float(mutual_info)
        }

    def stopping_criterion(self, X: np.ndarray, model, threshold: float = 0.01) -> bool:
        """
        Check if optimization should stop based on KG values.

        Args:
            X: Candidate points
            model: Current model
            threshold: KG threshold below which to stop

        Returns:
            True if optimization should stop
        """
        kg_values = self.evaluate(X, model)
        max_kg = np.max(kg_values)

        return max_kg < threshold

    @staticmethod
    def get_default_hyperparameters():
        """Get default hyperparameters for Knowledge Gradient"""
        return {
            'n_fantasy_points': [50, 100, 200, 500],
            'minimize': [True, False],
            'discrete_optimization': [True, False]
        }