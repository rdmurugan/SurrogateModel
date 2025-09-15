import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
from ..acquisition.factory import AcquisitionFunctionFactory


class FidelityLevel:
    """Represents a single fidelity level in multi-fidelity modeling"""

    def __init__(self, level: int, cost: float, accuracy: float,
                 model=None, name: str = None):
        """
        Initialize a fidelity level.

        Args:
            level: Fidelity level (0 = lowest, higher numbers = higher fidelity)
            cost: Relative cost of evaluation (1.0 = reference cost)
            accuracy: Expected accuracy level (0.0-1.0)
            model: Surrogate model for this fidelity
            name: Human-readable name for this fidelity
        """
        self.level = level
        self.cost = cost
        self.accuracy = accuracy
        self.model = model
        self.name = name or f"Fidelity_{level}"
        self.data_X = []
        self.data_y = []
        self.n_samples = 0

    def add_data(self, X: np.ndarray, y: np.ndarray):
        """Add training data to this fidelity level"""
        self.data_X.append(X)
        self.data_y.append(y)
        self.n_samples += len(X)

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all training data for this fidelity"""
        if not self.data_X:
            return np.array([]).reshape(0, -1), np.array([])

        X_all = np.vstack(self.data_X)
        y_all = np.hstack(self.data_y)
        return X_all, y_all


class MultiFidelityModel(ABC):
    """
    Abstract base class for multi-fidelity surrogate models.

    Multi-fidelity modeling combines information from multiple simulation
    fidelities to create more accurate surrogates with fewer high-fidelity
    evaluations.
    """

    def __init__(self, fidelity_levels: List[Dict[str, Any]]):
        """
        Initialize multi-fidelity model.

        Args:
            fidelity_levels: List of dictionaries defining each fidelity level
                           Each dict should contain: level, cost, accuracy, name
        """
        self.fidelity_levels = []

        for fid_info in fidelity_levels:
            fidelity = FidelityLevel(
                level=fid_info['level'],
                cost=fid_info['cost'],
                accuracy=fid_info['accuracy'],
                name=fid_info.get('name')
            )
            self.fidelity_levels.append(fidelity)

        # Sort by fidelity level
        self.fidelity_levels.sort(key=lambda x: x.level)

        self.is_trained = False
        self.acquisition_function = None

    @abstractmethod
    def fit(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Train the multi-fidelity model.

        Args:
            multi_fidelity_data: Dict mapping fidelity level to (X, y) data
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, fidelity_level: int = None) -> Dict[str, Any]:
        """
        Make predictions at specified fidelity level.

        Args:
            X: Input points
            fidelity_level: Target fidelity level (None for highest)

        Returns:
            Predictions with uncertainty
        """
        pass

    def add_data(self, X: np.ndarray, y: np.ndarray, fidelity_level: int):
        """Add new data to a specific fidelity level"""
        fidelity = self.get_fidelity_level(fidelity_level)
        fidelity.add_data(X, y)

    def get_fidelity_level(self, level: int) -> FidelityLevel:
        """Get fidelity level object by level number"""
        for fidelity in self.fidelity_levels:
            if fidelity.level == level:
                return fidelity
        raise ValueError(f"Fidelity level {level} not found")

    def get_highest_fidelity(self) -> FidelityLevel:
        """Get the highest fidelity level"""
        return max(self.fidelity_levels, key=lambda x: x.level)

    def get_lowest_fidelity(self) -> FidelityLevel:
        """Get the lowest fidelity level"""
        return min(self.fidelity_levels, key=lambda x: x.level)

    def get_data_summary(self) -> Dict[int, Dict[str, Any]]:
        """Get summary of data at each fidelity level"""
        summary = {}

        for fidelity in self.fidelity_levels:
            summary[fidelity.level] = {
                'name': fidelity.name,
                'n_samples': fidelity.n_samples,
                'cost': fidelity.cost,
                'accuracy': fidelity.accuracy
            }

        return summary

    def recommend_next_sample(self, X_candidates: np.ndarray,
                             acquisition_function: str = 'expected_improvement',
                             budget_remaining: float = None) -> Dict[str, Any]:
        """
        Recommend next sample considering multiple fidelities.

        Args:
            X_candidates: Candidate points to evaluate
            acquisition_function: Acquisition function to use
            budget_remaining: Remaining evaluation budget

        Returns:
            Recommendation with fidelity level and point
        """
        if not self.is_trained:
            # Start with lowest fidelity if no model trained
            lowest_fidelity = self.get_lowest_fidelity()

            # Simple random selection for initialization
            idx = np.random.randint(len(X_candidates))

            return {
                'point': X_candidates[idx],
                'fidelity_level': lowest_fidelity.level,
                'acquisition_value': 1.0,
                'reasoning': 'Initial exploration with lowest fidelity'
            }

        # Get acquisition function
        if isinstance(acquisition_function, str):
            acq_func = AcquisitionFunctionFactory.create(acquisition_function)
        else:
            acq_func = acquisition_function

        recommendations = []

        # Evaluate each fidelity level
        for fidelity in self.fidelity_levels:
            if fidelity.model is None:
                continue

            # Skip if budget constraint violated
            if budget_remaining is not None and fidelity.cost > budget_remaining:
                continue

            # Set current best for acquisition function
            if hasattr(fidelity.model, 'training_metrics'):
                # Extract best value from training data
                X_fid, y_fid = fidelity.get_all_data()
                if len(y_fid) > 0:
                    current_best = np.max(y_fid)  # Assuming maximization
                    acq_func.set_current_best(current_best)

            # Evaluate acquisition function
            acq_values = acq_func.evaluate(X_candidates, fidelity.model)

            # Cost-adjusted acquisition value
            cost_adjusted_acq = acq_values / fidelity.cost

            best_idx = np.argmax(cost_adjusted_acq)

            recommendations.append({
                'point': X_candidates[best_idx],
                'fidelity_level': fidelity.level,
                'acquisition_value': acq_values[best_idx],
                'cost_adjusted_value': cost_adjusted_acq[best_idx],
                'cost': fidelity.cost,
                'reasoning': f'Best point for fidelity {fidelity.level}'
            })

        # Select best cost-adjusted recommendation
        if recommendations:
            best_rec = max(recommendations, key=lambda x: x['cost_adjusted_value'])
            return best_rec
        else:
            # Fallback
            lowest_fidelity = self.get_lowest_fidelity()
            idx = np.random.randint(len(X_candidates))

            return {
                'point': X_candidates[idx],
                'fidelity_level': lowest_fidelity.level,
                'acquisition_value': 0.0,
                'reasoning': 'Fallback selection'
            }

    def optimize_fidelity_allocation(self, total_budget: float,
                                   X_candidates: np.ndarray,
                                   optimization_horizon: int = 10) -> List[Dict[str, Any]]:
        """
        Optimize allocation of samples across fidelity levels.

        Args:
            total_budget: Total evaluation budget
            X_candidates: Candidate points for evaluation
            optimization_horizon: Number of steps to plan ahead

        Returns:
            List of recommended evaluations
        """
        plan = []
        remaining_budget = total_budget

        for step in range(optimization_horizon):
            if remaining_budget <= 0:
                break

            # Get recommendation for current state
            recommendation = self.recommend_next_sample(
                X_candidates, budget_remaining=remaining_budget
            )

            if recommendation['cost'] <= remaining_budget:
                plan.append({
                    'step': step,
                    'point': recommendation['point'],
                    'fidelity_level': recommendation['fidelity_level'],
                    'cost': recommendation['cost'],
                    'expected_improvement': recommendation.get('acquisition_value', 0),
                    'reasoning': recommendation['reasoning']
                })

                remaining_budget -= recommendation['cost']

                # Simulate adding this data (simplified)
                # In practice, you'd update the model

        return plan

    def get_fidelity_correlation_matrix(self) -> np.ndarray:
        """
        Calculate correlation matrix between fidelity levels.

        This helps understand the relationship between different fidelities.
        """
        correlations = np.eye(len(self.fidelity_levels))

        # Get common evaluation points
        common_points = self._find_common_evaluation_points()

        if len(common_points) < 2:
            return correlations

        # Calculate correlations where data overlap exists
        for i, fid_i in enumerate(self.fidelity_levels):
            for j, fid_j in enumerate(self.fidelity_levels):
                if i != j:
                    corr = self._calculate_fidelity_correlation(fid_i, fid_j, common_points)
                    correlations[i, j] = corr

        return correlations

    def _find_common_evaluation_points(self) -> List[np.ndarray]:
        """Find points that have been evaluated at multiple fidelities"""
        # Simplified implementation - in practice this would be more sophisticated
        common_points = []

        # For now, return empty list - would need actual implementation
        # based on specific problem setup

        return common_points

    def _calculate_fidelity_correlation(self, fid_i: FidelityLevel,
                                      fid_j: FidelityLevel,
                                      common_points: List[np.ndarray]) -> float:
        """Calculate correlation between two fidelity levels"""
        # Simplified - would need actual correlation calculation
        # based on common evaluation points

        # Default assumption: higher fidelity levels are more correlated
        level_diff = abs(fid_i.level - fid_j.level)
        correlation = max(0.1, 1.0 - 0.2 * level_diff)

        return correlation

    def estimate_model_discrepancy(self) -> Dict[int, float]:
        """
        Estimate discrepancy between fidelity levels.

        Returns:
            Dict mapping fidelity level to estimated discrepancy from highest fidelity
        """
        discrepancies = {}
        highest_fidelity = self.get_highest_fidelity()

        for fidelity in self.fidelity_levels:
            if fidelity.level == highest_fidelity.level:
                discrepancies[fidelity.level] = 0.0
            else:
                # Simple estimate based on accuracy difference
                discrepancy = 1.0 - (fidelity.accuracy / highest_fidelity.accuracy)
                discrepancies[fidelity.level] = discrepancy

        return discrepancies

    def get_cost_effectiveness_analysis(self) -> Dict[int, float]:
        """
        Analyze cost-effectiveness of each fidelity level.

        Returns:
            Dict mapping fidelity level to cost-effectiveness score
        """
        effectiveness = {}

        for fidelity in self.fidelity_levels:
            # Cost-effectiveness = accuracy / cost
            ce_score = fidelity.accuracy / fidelity.cost if fidelity.cost > 0 else 0
            effectiveness[fidelity.level] = ce_score

        return effectiveness