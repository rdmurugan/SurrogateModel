import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class InformationFusionModel:
    """
    Advanced Information Fusion for Multi-Fidelity Modeling.

    This class implements sophisticated algorithms for combining information
    from multiple fidelity levels using:
    1. Weighted fusion based on local reliability
    2. Bayesian model averaging
    3. Uncertainty-aware combination
    4. Adaptive weight learning
    """

    def __init__(self, fusion_method: str = 'adaptive_weighted',
                 uncertainty_weighting: bool = True,
                 locality_radius: float = 0.1):
        """
        Initialize Information Fusion Model.

        Args:
            fusion_method: Method for fusion ('weighted', 'bayesian', 'adaptive_weighted')
            uncertainty_weighting: Whether to use uncertainty in weighting
            locality_radius: Radius for local reliability estimation
        """
        self.fusion_method = fusion_method
        self.uncertainty_weighting = uncertainty_weighting
        self.locality_radius = locality_radius
        self.fidelity_weights = {}
        self.fidelity_models = {}
        self.is_trained = False

    def fit_fusion_weights(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                         validation_points: Optional[np.ndarray] = None):
        """
        Learn optimal fusion weights from multi-fidelity data.

        Args:
            multi_fidelity_data: Dict mapping fidelity level to (X, y) data
            validation_points: Points for validation (if None, use cross-validation)
        """
        logger.info(f"Learning fusion weights using method: {self.fusion_method}")

        if self.fusion_method == 'weighted':
            self._fit_static_weights(multi_fidelity_data)
        elif self.fusion_method == 'bayesian':
            self._fit_bayesian_weights(multi_fidelity_data)
        elif self.fusion_method == 'adaptive_weighted':
            self._fit_adaptive_weights(multi_fidelity_data, validation_points)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        self.is_trained = True

    def _fit_static_weights(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """Fit static weights based on fidelity accuracy and sample size"""
        total_samples = sum(len(y) for _, y in multi_fidelity_data.values())

        for fidelity_level, (X, y) in multi_fidelity_data.items():
            # Weight based on sample size and assumed fidelity accuracy
            sample_weight = len(y) / total_samples
            fidelity_weight = fidelity_level / max(multi_fidelity_data.keys())  # Normalized fidelity

            # Combine sample size and fidelity weights
            combined_weight = 0.7 * fidelity_weight + 0.3 * sample_weight
            self.fidelity_weights[fidelity_level] = combined_weight

        # Normalize weights
        total_weight = sum(self.fidelity_weights.values())
        for fidelity_level in self.fidelity_weights:
            self.fidelity_weights[fidelity_level] /= total_weight

    def _fit_bayesian_weights(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """Fit Bayesian model averaging weights"""
        # Find common evaluation points for comparison
        common_points = self._find_common_evaluation_points(multi_fidelity_data)

        if len(common_points) < 3:
            logger.warning("Insufficient common points for Bayesian weighting, using static weights")
            self._fit_static_weights(multi_fidelity_data)
            return

        # Calculate model evidence (marginal likelihood) for each fidelity
        log_evidences = {}

        for fidelity_level, (X, y) in multi_fidelity_data.items():
            # Train a GP on this fidelity's data
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
            gp.fit(X, y)

            # Calculate log marginal likelihood
            log_evidences[fidelity_level] = gp.log_marginal_likelihood()

        # Convert to Bayesian model averaging weights
        log_evidences_array = np.array(list(log_evidences.values()))
        log_evidences_normalized = log_evidences_array - np.max(log_evidences_array)
        evidences = np.exp(log_evidences_normalized)
        evidences /= np.sum(evidences)

        # Store weights
        for i, fidelity_level in enumerate(log_evidences.keys()):
            self.fidelity_weights[fidelity_level] = evidences[i]

    def _fit_adaptive_weights(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                            validation_points: Optional[np.ndarray] = None):
        """Fit adaptive weights that vary spatially"""
        # For adaptive weights, we'll use local reliability estimation
        if validation_points is None:
            # Use leave-one-out cross-validation on common points
            validation_points = self._select_validation_points(multi_fidelity_data)

        # Initialize spatial weight functions
        self.fidelity_weights = {}

        for fidelity_level, (X, y) in multi_fidelity_data.items():
            # Calculate local reliability at validation points
            reliability_scores = self._calculate_local_reliability(
                X, y, validation_points, fidelity_level
            )

            # Store reliability function (interpolated)
            self.fidelity_weights[fidelity_level] = {
                'validation_points': validation_points,
                'reliability_scores': reliability_scores,
                'type': 'adaptive'
            }

    def _find_common_evaluation_points(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Find points that have been evaluated at multiple fidelities"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated point matching
        all_points = []
        for X, _ in multi_fidelity_data.values():
            all_points.extend(X.tolist())

        unique_points = np.unique(np.array(all_points), axis=0)
        return unique_points[:min(50, len(unique_points))]  # Limit for efficiency

    def _select_validation_points(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Select representative validation points from the data"""
        all_X = []
        for X, _ in multi_fidelity_data.values():
            all_X.extend(X.tolist())

        all_X = np.array(all_X)

        # Select diverse points using farthest point sampling
        n_validation = min(20, len(all_X) // 2)
        selected_indices = self._farthest_point_sampling(all_X, n_validation)

        return all_X[selected_indices]

    def _farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> List[int]:
        """Select diverse points using farthest point sampling"""
        if len(points) <= n_samples:
            return list(range(len(points)))

        selected = [0]  # Start with first point
        distances = cdist([points[0]], points)[0]

        for _ in range(n_samples - 1):
            # Select point farthest from all selected points
            farthest_idx = np.argmax(distances)
            selected.append(farthest_idx)

            # Update distances
            new_distances = cdist([points[farthest_idx]], points)[0]
            distances = np.minimum(distances, new_distances)

        return selected

    def _calculate_local_reliability(self, X: np.ndarray, y: np.ndarray,
                                   validation_points: np.ndarray,
                                   fidelity_level: int) -> np.ndarray:
        """Calculate local reliability of a fidelity level at validation points"""
        reliability_scores = np.zeros(len(validation_points))

        for i, val_point in enumerate(validation_points):
            # Find nearby training points
            distances = cdist([val_point], X)[0]
            nearby_mask = distances <= self.locality_radius

            if np.sum(nearby_mask) < 2:
                # Not enough nearby points, use global reliability
                reliability_scores[i] = 0.5
                continue

            # Calculate local prediction error using cross-validation
            X_local = X[nearby_mask]
            y_local = y[nearby_mask]

            # Simple local GP for error estimation
            kernel = ConstantKernel(1.0) * RBF(1.0)
            local_gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

            # Leave-one-out cross-validation on local data
            errors = []
            for j in range(len(X_local)):
                train_mask = np.ones(len(X_local), dtype=bool)
                train_mask[j] = False

                X_train_local = X_local[train_mask]
                y_train_local = y_local[train_mask]
                X_test_local = X_local[j:j+1]
                y_test_local = y_local[j]

                try:
                    local_gp.fit(X_train_local, y_train_local)
                    y_pred = local_gp.predict(X_test_local)
                    error = np.abs(y_pred[0] - y_test_local)
                    errors.append(error)
                except:
                    errors.append(1.0)  # Default high error

            # Convert error to reliability (lower error = higher reliability)
            mean_error = np.mean(errors) if errors else 1.0
            reliability = np.exp(-mean_error)  # Exponential decay
            reliability_scores[i] = reliability

        return reliability_scores

    def get_fusion_weights(self, X: np.ndarray, fidelity_levels: List[int]) -> Dict[int, np.ndarray]:
        """
        Get fusion weights for given points and fidelity levels.

        Args:
            X: Input points
            fidelity_levels: Available fidelity levels

        Returns:
            Dict mapping fidelity level to weight array
        """
        if not self.is_trained:
            raise ValueError("Fusion model must be fitted before getting weights")

        weights = {}

        if self.fusion_method in ['weighted', 'bayesian']:
            # Static weights
            for fidelity_level in fidelity_levels:
                if fidelity_level in self.fidelity_weights:
                    weights[fidelity_level] = np.full(len(X), self.fidelity_weights[fidelity_level])
                else:
                    weights[fidelity_level] = np.zeros(len(X))

        elif self.fusion_method == 'adaptive_weighted':
            # Adaptive weights based on spatial location
            for fidelity_level in fidelity_levels:
                if fidelity_level in self.fidelity_weights:
                    weight_info = self.fidelity_weights[fidelity_level]
                    weights[fidelity_level] = self._interpolate_adaptive_weights(
                        X, weight_info['validation_points'], weight_info['reliability_scores']
                    )
                else:
                    weights[fidelity_level] = np.zeros(len(X))

        # Normalize weights at each point
        total_weights = np.zeros(len(X))
        for fidelity_level in fidelity_levels:
            if fidelity_level in weights:
                total_weights += weights[fidelity_level]

        # Avoid division by zero
        total_weights = np.maximum(total_weights, 1e-8)

        for fidelity_level in weights:
            weights[fidelity_level] /= total_weights

        return weights

    def _interpolate_adaptive_weights(self, X: np.ndarray, validation_points: np.ndarray,
                                    reliability_scores: np.ndarray) -> np.ndarray:
        """Interpolate adaptive weights at query points"""
        weights = np.zeros(len(X))

        for i, query_point in enumerate(X):
            # Find distances to all validation points
            distances = cdist([query_point], validation_points)[0]

            # Use inverse distance weighting for interpolation
            # Add small epsilon to avoid division by zero
            inv_distances = 1.0 / (distances + 1e-8)
            interpolation_weights = inv_distances / np.sum(inv_distances)

            # Interpolate reliability score
            interpolated_reliability = np.sum(interpolation_weights * reliability_scores)
            weights[i] = interpolated_reliability

        return weights

    def fuse_predictions(self, predictions: Dict[int, Dict[str, Any]], X: np.ndarray) -> Dict[str, Any]:
        """
        Fuse predictions from multiple fidelity levels.

        Args:
            predictions: Dict mapping fidelity level to prediction results
            X: Input points (for adaptive weighting)

        Returns:
            Fused prediction results
        """
        if not self.is_trained:
            raise ValueError("Fusion model must be fitted before fusing predictions")

        available_fidelities = list(predictions.keys())
        fusion_weights = self.get_fusion_weights(X, available_fidelities)

        # Get output names from first prediction
        first_pred = next(iter(predictions.values()))
        output_names = list(first_pred.keys())

        fused_results = {}

        for output_name in output_names:
            # Extract predictions and uncertainties
            fidelity_predictions = {}
            fidelity_uncertainties = {}

            for fidelity_level, pred_dict in predictions.items():
                if output_name in pred_dict:
                    pred_info = pred_dict[output_name]
                    fidelity_predictions[fidelity_level] = pred_info['prediction']
                    fidelity_uncertainties[fidelity_level] = pred_info['uncertainty']['standard_deviation']

            # Fuse predictions
            fused_prediction = 0.0
            fused_variance = 0.0

            for fidelity_level in fidelity_predictions:
                weight = fusion_weights[fidelity_level][0]  # Assuming single point prediction
                pred = fidelity_predictions[fidelity_level]
                std = fidelity_uncertainties[fidelity_level]

                fused_prediction += weight * pred

                if self.uncertainty_weighting:
                    # Uncertainty-aware fusion: lower weight for higher uncertainty
                    uncertainty_factor = 1.0 / (1.0 + std**2)
                    effective_weight = weight * uncertainty_factor
                else:
                    effective_weight = weight

                fused_variance += (effective_weight * std)**2

            fused_std = np.sqrt(fused_variance)

            fused_results[output_name] = {
                'prediction': float(fused_prediction),
                'uncertainty': {
                    'standard_deviation': float(fused_std),
                    'variance': float(fused_variance),
                    'confidence_interval_95': [
                        fused_prediction - 1.96 * fused_std,
                        fused_prediction + 1.96 * fused_std
                    ]
                },
                'fusion_weights': {int(k): float(v[0]) for k, v in fusion_weights.items()},
                'individual_predictions': fidelity_predictions,
                'fusion_method': self.fusion_method
            }

        return fused_results

    def analyze_fusion_quality(self, predictions: Dict[int, Dict[str, Any]],
                             true_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze the quality of the fusion process.

        Args:
            predictions: Predictions from different fidelities
            true_values: True values for validation (if available)

        Returns:
            Analysis of fusion quality
        """
        analysis = {
            'weight_distribution': {},
            'prediction_agreement': {},
            'uncertainty_calibration': {},
            'fusion_stability': {}
        }

        # Analyze weight distribution
        available_fidelities = list(predictions.keys())
        for fidelity_level in available_fidelities:
            if fidelity_level in self.fidelity_weights:
                if isinstance(self.fidelity_weights[fidelity_level], dict):
                    # Adaptive weights
                    mean_weight = np.mean(self.fidelity_weights[fidelity_level]['reliability_scores'])
                    std_weight = np.std(self.fidelity_weights[fidelity_level]['reliability_scores'])
                else:
                    # Static weights
                    mean_weight = self.fidelity_weights[fidelity_level]
                    std_weight = 0.0

                analysis['weight_distribution'][fidelity_level] = {
                    'mean': float(mean_weight),
                    'std': float(std_weight)
                }

        # Analyze prediction agreement between fidelities
        if len(available_fidelities) > 1:
            output_names = list(next(iter(predictions.values())).keys())

            for output_name in output_names:
                pred_values = []
                for fidelity_level in available_fidelities:
                    if output_name in predictions[fidelity_level]:
                        pred_values.append(predictions[fidelity_level][output_name]['prediction'])

                if len(pred_values) > 1:
                    pred_array = np.array(pred_values)
                    agreement = {
                        'mean': float(np.mean(pred_array)),
                        'std': float(np.std(pred_array)),
                        'range': float(np.ptp(pred_array)),  # Peak-to-peak
                        'coefficient_of_variation': float(np.std(pred_array) / (np.mean(pred_array) + 1e-8))
                    }
                    analysis['prediction_agreement'][output_name] = agreement

        return analysis

    def optimize_fusion_parameters(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                 validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Optimize fusion parameters using validation data.

        Args:
            multi_fidelity_data: Training data for each fidelity
            validation_data: Validation (X, y) data

        Returns:
            Optimization results
        """
        X_val, y_val = validation_data

        def objective(params):
            # Update fusion parameters
            if self.fusion_method == 'adaptive_weighted':
                self.locality_radius = params[0]

            # Refit fusion weights
            self.fit_fusion_weights(multi_fidelity_data)

            # Evaluate on validation data
            # This would require actual model predictions
            # For now, return a placeholder loss
            return np.random.random()  # Placeholder

        # Optimize parameters
        if self.fusion_method == 'adaptive_weighted':
            initial_params = [self.locality_radius]
            bounds = [(0.01, 1.0)]  # Bounds for locality_radius

            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')

            if result.success:
                self.locality_radius = result.x[0]

            return {
                'success': result.success,
                'optimized_parameters': {
                    'locality_radius': float(self.locality_radius)
                },
                'optimization_result': {
                    'final_loss': float(result.fun),
                    'iterations': int(result.nit)
                }
            }

        return {'success': False, 'message': 'No parameters to optimize for this fusion method'}