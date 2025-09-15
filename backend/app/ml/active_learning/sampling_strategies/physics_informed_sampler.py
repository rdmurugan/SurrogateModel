import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import warnings
from .base_sampler import BaseSampler


class PhysicsInformedSampler(BaseSampler):
    """
    Physics-informed sampling strategy for active learning.

    This sampler incorporates domain knowledge about the underlying physics
    to guide sampling decisions. It's particularly useful for engineering
    simulations where physical constraints and governing equations can
    inform optimal sampling strategies.

    Key strategies:
    1. Boundary-aware sampling: Focus on regions near boundaries/constraints
    2. Gradient-informed sampling: Sample where gradients are steep
    3. Conservation-aware sampling: Respect physical conservation laws
    4. Multi-physics coupling: Handle coupled physical phenomena
    5. Symmetry exploitation: Leverage problem symmetries
    """

    def __init__(self,
                 physics_constraints: Dict[str, Any] = None,
                 boundary_weights: Dict[str, float] = None,
                 conservation_laws: List[str] = None,
                 symmetries: List[Dict[str, Any]] = None,
                 coupling_strength: float = 0.5):
        """
        Initialize physics-informed sampler.

        Args:
            physics_constraints: Dictionary defining physical constraints
            boundary_weights: Weights for different boundary regions
            conservation_laws: List of conservation laws to respect
            symmetries: Problem symmetries to exploit
            coupling_strength: Strength of multi-physics coupling (0-1)
        """
        super().__init__()

        self.physics_constraints = physics_constraints or {}
        self.boundary_weights = boundary_weights or {}
        self.conservation_laws = conservation_laws or []
        self.symmetries = symmetries or []
        self.coupling_strength = coupling_strength

        # Physics-aware components
        self.boundary_detector = BoundaryDetector()
        self.gradient_estimator = GradientEstimator()
        self.conservation_checker = ConservationChecker(conservation_laws)
        self.symmetry_exploiter = SymmetryExploiter(symmetries)

        # Sampling history
        self.physics_scores = []
        self.boundary_violations = []

    def sample(self,
               model,
               X_candidates: np.ndarray,
               n_samples: int = 1,
               acquisition_function=None,
               **kwargs) -> Dict[str, Any]:
        """
        Sample points using physics-informed strategy.

        Args:
            model: Trained surrogate model
            X_candidates: Candidate sampling points
            n_samples: Number of points to sample
            acquisition_function: Base acquisition function to enhance

        Returns:
            Sampling results with physics insights
        """
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have predict method")

        if len(X_candidates) == 0:
            raise ValueError("No candidate points provided")

        # Get base acquisition scores if provided
        if acquisition_function is not None:
            base_scores = acquisition_function.evaluate(X_candidates, model)
        else:
            # Use uncertainty as default
            predictions = model.predict(X_candidates)
            if isinstance(predictions, dict):
                # Extract uncertainty from structured predictions
                base_scores = np.array([
                    pred.get('uncertainty', {}).get('standard_deviation', 0.1)
                    for pred in predictions.values()
                ])
            else:
                base_scores = np.ones(len(X_candidates))

        # Calculate physics-informed scores
        physics_scores = self._calculate_physics_scores(X_candidates, model)

        # Combine base and physics scores
        combined_scores = self._combine_scores(base_scores, physics_scores)

        # Select points using physics-aware selection
        selected_indices = self._physics_aware_selection(
            X_candidates, combined_scores, n_samples
        )

        selected_points = X_candidates[selected_indices]

        # Calculate physics insights for selected points
        insights = self._analyze_physics_insights(selected_points, model)

        return {
            'selected_points': selected_points,
            'selected_indices': selected_indices,
            'acquisition_scores': combined_scores[selected_indices],
            'physics_scores': physics_scores[selected_indices],
            'physics_insights': insights,
            'boundary_analysis': self._analyze_boundary_proximity(selected_points),
            'conservation_status': self._check_conservation_laws(selected_points),
            'symmetry_exploitation': self._analyze_symmetry_usage(selected_points)
        }

    def _calculate_physics_scores(self, X_candidates: np.ndarray, model) -> np.ndarray:
        """Calculate physics-informed scoring for candidates"""
        n_candidates = len(X_candidates)
        physics_scores = np.zeros(n_candidates)

        # 1. Boundary proximity scoring
        boundary_scores = self._score_boundary_proximity(X_candidates)

        # 2. Gradient-based scoring
        gradient_scores = self._score_gradients(X_candidates, model)

        # 3. Conservation law scoring
        conservation_scores = self._score_conservation_laws(X_candidates)

        # 4. Multi-physics coupling scoring
        coupling_scores = self._score_physics_coupling(X_candidates, model)

        # 5. Symmetry-based scoring
        symmetry_scores = self._score_symmetries(X_candidates)

        # Combine physics scores with weights
        weights = {
            'boundary': 0.25,
            'gradient': 0.25,
            'conservation': 0.2,
            'coupling': 0.15,
            'symmetry': 0.15
        }

        physics_scores = (
            weights['boundary'] * boundary_scores +
            weights['gradient'] * gradient_scores +
            weights['conservation'] * conservation_scores +
            weights['coupling'] * coupling_scores +
            weights['symmetry'] * symmetry_scores
        )

        return physics_scores

    def _score_boundary_proximity(self, X_candidates: np.ndarray) -> np.ndarray:
        """Score points based on proximity to physical boundaries"""
        scores = np.ones(len(X_candidates))

        # Detect boundaries in the design space
        boundaries = self.boundary_detector.detect_boundaries(
            X_candidates, self.physics_constraints
        )

        for boundary_name, boundary_info in boundaries.items():
            weight = self.boundary_weights.get(boundary_name, 1.0)

            # Calculate distance to boundary
            distances = boundary_info['distance_function'](X_candidates)

            # Higher scores for points near boundaries (but not too close)
            optimal_distance = boundary_info.get('optimal_distance', 0.1)
            boundary_scores = np.exp(-((distances - optimal_distance) ** 2) / (2 * optimal_distance ** 2))

            scores += weight * boundary_scores

        return scores / len(boundaries) if boundaries else scores

    def _score_gradients(self, X_candidates: np.ndarray, model) -> np.ndarray:
        """Score points based on predicted gradient information"""
        try:
            # Estimate gradients using finite differences
            gradients = self.gradient_estimator.estimate_gradients(X_candidates, model)

            # Higher scores where gradients are steep (high uncertainty regions)
            gradient_magnitudes = np.linalg.norm(gradients, axis=1)

            # Normalize scores
            if np.max(gradient_magnitudes) > 0:
                scores = gradient_magnitudes / np.max(gradient_magnitudes)
            else:
                scores = np.ones(len(X_candidates))

            return scores

        except Exception as e:
            warnings.warn(f"Gradient estimation failed: {e}")
            return np.ones(len(X_candidates))

    def _score_conservation_laws(self, X_candidates: np.ndarray) -> np.ndarray:
        """Score points based on conservation law compliance"""
        scores = np.ones(len(X_candidates))

        for law in self.conservation_laws:
            law_scores = self.conservation_checker.check_conservation(
                X_candidates, law, self.physics_constraints
            )
            scores *= law_scores

        return scores

    def _score_physics_coupling(self, X_candidates: np.ndarray, model) -> np.ndarray:
        """Score points based on multi-physics coupling strength"""
        if self.coupling_strength == 0:
            return np.ones(len(X_candidates))

        # Estimate coupling strength based on model predictions
        try:
            predictions = model.predict(X_candidates)

            # For multi-output models, look at correlation between outputs
            if hasattr(model, 'n_outputs_') and model.n_outputs_ > 1:
                # Calculate coupling based on output correlations
                coupling_scores = self._estimate_coupling_strength(predictions)
            else:
                # Single output - use variance as proxy for coupling complexity
                if isinstance(predictions, dict):
                    variances = np.array([
                        pred.get('uncertainty', {}).get('variance', 0.1)
                        for pred in predictions.values()
                    ])
                else:
                    variances = np.ones(len(X_candidates))

                coupling_scores = variances / np.max(variances) if np.max(variances) > 0 else variances

            return coupling_scores

        except Exception as e:
            warnings.warn(f"Coupling estimation failed: {e}")
            return np.ones(len(X_candidates))

    def _score_symmetries(self, X_candidates: np.ndarray) -> np.ndarray:
        """Score points based on symmetry exploitation"""
        if not self.symmetries:
            return np.ones(len(X_candidates))

        return self.symmetry_exploiter.score_symmetry_value(
            X_candidates, self.symmetries
        )

    def _combine_scores(self, base_scores: np.ndarray, physics_scores: np.ndarray) -> np.ndarray:
        """Combine base acquisition scores with physics scores"""
        # Normalize both score arrays
        base_norm = base_scores / np.max(base_scores) if np.max(base_scores) > 0 else base_scores
        physics_norm = physics_scores / np.max(physics_scores) if np.max(physics_scores) > 0 else physics_scores

        # Weighted combination
        alpha = 0.6  # Weight for base acquisition
        beta = 0.4   # Weight for physics information

        combined = alpha * base_norm + beta * physics_norm

        # Store for analysis
        self.physics_scores.append(physics_scores)

        return combined

    def _physics_aware_selection(self,
                                X_candidates: np.ndarray,
                                scores: np.ndarray,
                                n_samples: int) -> np.ndarray:
        """Select points using physics-aware strategy"""
        if n_samples == 1:
            return np.array([np.argmax(scores)])

        selected_indices = []
        remaining_indices = np.arange(len(X_candidates))

        for i in range(n_samples):
            if len(remaining_indices) == 0:
                break

            # Select point with highest score among remaining
            relative_scores = scores[remaining_indices]
            best_relative_idx = np.argmax(relative_scores)
            best_idx = remaining_indices[best_relative_idx]

            selected_indices.append(best_idx)

            # Update remaining candidates considering physics constraints
            remaining_indices = self._update_remaining_candidates(
                remaining_indices, best_idx, X_candidates
            )

        return np.array(selected_indices)

    def _update_remaining_candidates(self,
                                   remaining_indices: np.ndarray,
                                   selected_idx: int,
                                   X_candidates: np.ndarray) -> np.ndarray:
        """Update remaining candidates based on physics constraints"""
        # Remove selected point
        remaining_indices = remaining_indices[remaining_indices != selected_idx]

        # Apply physics-based filtering
        selected_point = X_candidates[selected_idx:selected_idx+1]

        # Remove points that violate minimum distance constraints
        if 'min_physics_distance' in self.physics_constraints:
            min_dist = self.physics_constraints['min_physics_distance']
            distances = cdist(selected_point, X_candidates[remaining_indices])
            valid_mask = distances[0] >= min_dist
            remaining_indices = remaining_indices[valid_mask]

        return remaining_indices

    def _analyze_physics_insights(self, selected_points: np.ndarray, model) -> Dict[str, Any]:
        """Analyze physics insights for selected points"""
        insights = {
            'boundary_proximity': {},
            'gradient_analysis': {},
            'conservation_compliance': {},
            'coupling_analysis': {},
            'symmetry_usage': {}
        }

        # Boundary analysis
        boundaries = self.boundary_detector.detect_boundaries(
            selected_points, self.physics_constraints
        )

        for boundary_name, boundary_info in boundaries.items():
            distances = boundary_info['distance_function'](selected_points)
            insights['boundary_proximity'][boundary_name] = {
                'mean_distance': float(np.mean(distances)),
                'min_distance': float(np.min(distances)),
                'coverage': len(distances[distances < boundary_info.get('influence_radius', 0.2)])
            }

        # Gradient analysis
        try:
            gradients = self.gradient_estimator.estimate_gradients(selected_points, model)
            gradient_magnitudes = np.linalg.norm(gradients, axis=1)

            insights['gradient_analysis'] = {
                'mean_gradient_magnitude': float(np.mean(gradient_magnitudes)),
                'max_gradient_magnitude': float(np.max(gradient_magnitudes)),
                'gradient_diversity': float(np.std(gradient_magnitudes))
            }
        except Exception:
            insights['gradient_analysis'] = {'error': 'Gradient estimation failed'}

        return insights

    def _analyze_boundary_proximity(self, selected_points: np.ndarray) -> Dict[str, Any]:
        """Analyze boundary proximity for selected points"""
        analysis = {}

        boundaries = self.boundary_detector.detect_boundaries(
            selected_points, self.physics_constraints
        )

        for boundary_name, boundary_info in boundaries.items():
            distances = boundary_info['distance_function'](selected_points)

            analysis[boundary_name] = {
                'points_near_boundary': int(np.sum(distances < 0.1)),
                'average_distance': float(np.mean(distances)),
                'boundary_coverage_score': float(np.exp(-np.mean(distances)))
            }

        return analysis

    def _check_conservation_laws(self, selected_points: np.ndarray) -> Dict[str, Any]:
        """Check conservation law compliance for selected points"""
        status = {}

        for law in self.conservation_laws:
            compliance = self.conservation_checker.check_conservation(
                selected_points, law, self.physics_constraints
            )

            status[law] = {
                'compliance_rate': float(np.mean(compliance)),
                'violations': int(np.sum(compliance < 0.9)),
                'average_compliance': float(np.mean(compliance))
            }

        return status

    def _analyze_symmetry_usage(self, selected_points: np.ndarray) -> Dict[str, Any]:
        """Analyze symmetry exploitation in selected points"""
        if not self.symmetries:
            return {}

        return self.symmetry_exploiter.analyze_symmetry_usage(
            selected_points, self.symmetries
        )

    def _estimate_coupling_strength(self, predictions) -> np.ndarray:
        """Estimate multi-physics coupling strength"""
        # Simplified coupling estimation
        # In practice, this would be more sophisticated

        if isinstance(predictions, dict):
            # Extract multiple outputs
            outputs = []
            for pred in predictions.values():
                if isinstance(pred, dict) and 'prediction' in pred:
                    outputs.append(pred['prediction'])

            if len(outputs) > 1:
                # Calculate correlation between outputs
                correlations = np.corrcoef(outputs)
                coupling_strength = np.mean(np.abs(correlations))
                return np.full(len(predictions), coupling_strength)

        return np.ones(len(predictions))

    def get_physics_summary(self) -> Dict[str, Any]:
        """Get summary of physics-informed sampling performance"""
        if not self.physics_scores:
            return {}

        physics_scores_array = np.vstack(self.physics_scores)

        return {
            'average_physics_score': float(np.mean(physics_scores_array)),
            'physics_score_trend': [float(np.mean(scores)) for scores in self.physics_scores],
            'boundary_weight_usage': dict(self.boundary_weights),
            'conservation_laws_active': self.conservation_laws,
            'symmetries_exploited': len(self.symmetries),
            'coupling_strength': float(self.coupling_strength)
        }


class BoundaryDetector:
    """Detect and analyze physical boundaries in the design space"""

    def detect_boundaries(self, X: np.ndarray, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Detect boundaries based on physics constraints"""
        boundaries = {}

        # Domain boundaries
        if 'domain_bounds' in constraints:
            bounds = constraints['domain_bounds']
            boundaries['domain'] = self._create_domain_boundary(bounds)

        # Physical constraint boundaries
        if 'physical_constraints' in constraints:
            for constraint_name, constraint_info in constraints['physical_constraints'].items():
                boundaries[constraint_name] = self._create_constraint_boundary(constraint_info)

        # Material interface boundaries
        if 'material_interfaces' in constraints:
            for interface_name, interface_info in constraints['material_interfaces'].items():
                boundaries[interface_name] = self._create_interface_boundary(interface_info)

        return boundaries

    def _create_domain_boundary(self, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Create domain boundary detection function"""
        def distance_function(X: np.ndarray) -> np.ndarray:
            distances = np.zeros(len(X))
            for i, (lower, upper) in enumerate(bounds):
                if i < X.shape[1]:
                    dist_to_lower = X[:, i] - lower
                    dist_to_upper = upper - X[:, i]
                    distances += np.minimum(dist_to_lower, dist_to_upper)
            return distances / len(bounds)

        return {
            'distance_function': distance_function,
            'optimal_distance': 0.1,
            'influence_radius': 0.2
        }

    def _create_constraint_boundary(self, constraint_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create physics constraint boundary"""
        constraint_type = constraint_info.get('type', 'inequality')

        if constraint_type == 'inequality':
            def distance_function(X: np.ndarray) -> np.ndarray:
                # Simplified - would implement specific constraint
                return np.ones(len(X)) * 0.1
        else:
            def distance_function(X: np.ndarray) -> np.ndarray:
                return np.ones(len(X)) * 0.1

        return {
            'distance_function': distance_function,
            'optimal_distance': constraint_info.get('optimal_distance', 0.05),
            'influence_radius': constraint_info.get('influence_radius', 0.15)
        }

    def _create_interface_boundary(self, interface_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create material interface boundary"""
        def distance_function(X: np.ndarray) -> np.ndarray:
            # Simplified interface detection
            return np.ones(len(X)) * 0.1

        return {
            'distance_function': distance_function,
            'optimal_distance': 0.02,
            'influence_radius': 0.1
        }


class GradientEstimator:
    """Estimate gradients for physics-informed sampling"""

    def estimate_gradients(self, X: np.ndarray, model) -> np.ndarray:
        """Estimate gradients using finite differences"""
        n_points, n_dims = X.shape
        gradients = np.zeros((n_points, n_dims))

        eps = 1e-6

        for i in range(n_points):
            for j in range(n_dims):
                # Forward difference
                X_forward = X[i:i+1].copy()
                X_forward[0, j] += eps

                X_backward = X[i:i+1].copy()
                X_backward[0, j] -= eps

                # Get predictions
                pred_forward = self._get_prediction_value(model.predict(X_forward))
                pred_backward = self._get_prediction_value(model.predict(X_backward))

                # Calculate gradient
                gradients[i, j] = (pred_forward - pred_backward) / (2 * eps)

        return gradients

    def _get_prediction_value(self, prediction) -> float:
        """Extract scalar prediction value"""
        if isinstance(prediction, dict):
            # Extract from structured prediction
            for key, value in prediction.items():
                if isinstance(value, dict) and 'prediction' in value:
                    return float(value['prediction'])
            return 0.0
        elif isinstance(prediction, np.ndarray):
            return float(prediction.flatten()[0])
        else:
            return float(prediction)


class ConservationChecker:
    """Check conservation law compliance"""

    def __init__(self, conservation_laws: List[str]):
        self.conservation_laws = conservation_laws

    def check_conservation(self, X: np.ndarray, law: str, constraints: Dict[str, Any]) -> np.ndarray:
        """Check conservation law compliance for points"""
        if law == 'mass_conservation':
            return self._check_mass_conservation(X, constraints)
        elif law == 'energy_conservation':
            return self._check_energy_conservation(X, constraints)
        elif law == 'momentum_conservation':
            return self._check_momentum_conservation(X, constraints)
        else:
            return np.ones(len(X))

    def _check_mass_conservation(self, X: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Check mass conservation compliance"""
        # Simplified implementation
        return np.ones(len(X))

    def _check_energy_conservation(self, X: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Check energy conservation compliance"""
        # Simplified implementation
        return np.ones(len(X))

    def _check_momentum_conservation(self, X: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Check momentum conservation compliance"""
        # Simplified implementation
        return np.ones(len(X))


class SymmetryExploiter:
    """Exploit problem symmetries for efficient sampling"""

    def __init__(self, symmetries: List[Dict[str, Any]]):
        self.symmetries = symmetries

    def score_symmetry_value(self, X: np.ndarray, symmetries: List[Dict[str, Any]]) -> np.ndarray:
        """Score points based on symmetry exploitation value"""
        if not symmetries:
            return np.ones(len(X))

        scores = np.zeros(len(X))

        for symmetry in symmetries:
            symmetry_type = symmetry.get('type', 'reflection')

            if symmetry_type == 'reflection':
                scores += self._score_reflection_symmetry(X, symmetry)
            elif symmetry_type == 'rotation':
                scores += self._score_rotational_symmetry(X, symmetry)
            elif symmetry_type == 'translation':
                scores += self._score_translational_symmetry(X, symmetry)

        return scores / len(symmetries) if symmetries else np.ones(len(X))

    def _score_reflection_symmetry(self, X: np.ndarray, symmetry: Dict[str, Any]) -> np.ndarray:
        """Score reflection symmetry exploitation"""
        axis = symmetry.get('axis', 0)
        reflection_plane = symmetry.get('plane_location', 0.5)

        # Higher scores for points that can provide information about symmetric regions
        distances_to_plane = np.abs(X[:, axis] - reflection_plane)
        scores = np.exp(-distances_to_plane)

        return scores

    def _score_rotational_symmetry(self, X: np.ndarray, symmetry: Dict[str, Any]) -> np.ndarray:
        """Score rotational symmetry exploitation"""
        # Simplified implementation
        return np.ones(len(X))

    def _score_translational_symmetry(self, X: np.ndarray, symmetry: Dict[str, Any]) -> np.ndarray:
        """Score translational symmetry exploitation"""
        # Simplified implementation
        return np.ones(len(X))

    def analyze_symmetry_usage(self, X: np.ndarray, symmetries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well symmetries are being exploited"""
        analysis = {}

        for i, symmetry in enumerate(symmetries):
            symmetry_name = symmetry.get('name', f'symmetry_{i}')
            symmetry_type = symmetry.get('type', 'unknown')

            analysis[symmetry_name] = {
                'type': symmetry_type,
                'exploitation_score': float(np.mean(self.score_symmetry_value(X, [symmetry]))),
                'coverage': self._calculate_symmetry_coverage(X, symmetry)
            }

        return analysis

    def _calculate_symmetry_coverage(self, X: np.ndarray, symmetry: Dict[str, Any]) -> float:
        """Calculate how well the symmetry is covered by sample points"""
        # Simplified coverage calculation
        return 0.8  # Placeholder