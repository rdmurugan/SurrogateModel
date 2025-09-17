import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.ensemble import RandomForestRegressor
from .multi_fidelity_model import MultiFidelityModel, FidelityLevel
import logging

logger = logging.getLogger(__name__)


class HierarchicalMultiFidelityModel(MultiFidelityModel):
    """
    Hierarchical Multi-Fidelity Model using recursive modeling approach.

    This implementation creates a hierarchy of models where each fidelity level
    is modeled as a correction to the lower fidelity levels, enabling efficient
    information propagation across the fidelity spectrum.

    The hierarchical approach:
    1. f_0(x) ~ GP(μ_0, k_0)  [Lowest fidelity]
    2. f_1(x) = f_0(x) + δ_1(x, f_0(x))  [Next fidelity]
    3. f_i(x) = f_{i-1}(x) + δ_i(x, f_{i-1}(x))  [Recursive structure]

    This enables modeling complex non-linear relationships between fidelities
    and better uncertainty propagation.
    """

    def __init__(self, fidelity_levels: List[Dict[str, Any]],
                 base_kernel: str = 'rbf',
                 use_input_augmentation: bool = True,
                 correlation_threshold: float = 0.3):
        """
        Initialize Hierarchical Multi-Fidelity Model.

        Args:
            fidelity_levels: List of fidelity level definitions
            base_kernel: Base kernel for GPs ('rbf', 'matern52', 'matern32')
            use_input_augmentation: Whether to augment inputs with lower fidelity predictions
            correlation_threshold: Minimum correlation to use hierarchical structure
        """
        super().__init__(fidelity_levels)
        self.base_kernel = base_kernel
        self.use_input_augmentation = use_input_augmentation
        self.correlation_threshold = correlation_threshold
        self.fidelity_models = {}  # Store trained models for each fidelity
        self.fidelity_corrections = {}  # Store correction models
        self.fidelity_correlations = {}  # Store correlation analysis

    def _create_kernel(self, input_dim: int):
        """Create appropriate kernel based on specification"""
        if self.base_kernel == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=np.ones(input_dim))
        elif self.base_kernel == 'matern52':
            return ConstantKernel(1.0) * Matern(length_scale=np.ones(input_dim), nu=2.5)
        elif self.base_kernel == 'matern32':
            return ConstantKernel(1.0) * Matern(length_scale=np.ones(input_dim), nu=1.5)
        else:
            return ConstantKernel(1.0) * RBF(length_scale=np.ones(input_dim))

    def fit(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Train the hierarchical multi-fidelity model.

        Args:
            multi_fidelity_data: Dict mapping fidelity level to (X, y) data
        """
        logger.info("Training Hierarchical Multi-Fidelity Model")

        # Sort fidelities by level
        sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)

        # Train base model (lowest fidelity)
        base_fidelity = sorted_fidelities[0]
        if base_fidelity.level not in multi_fidelity_data:
            raise ValueError(f"No data provided for base fidelity level {base_fidelity.level}")

        X_base, y_base = multi_fidelity_data[base_fidelity.level]
        self._train_base_model(X_base, y_base, base_fidelity.level)

        # Add data to fidelity level
        base_fidelity.add_data(X_base, y_base)

        # Train hierarchical corrections for higher fidelities
        for i, fidelity in enumerate(sorted_fidelities[1:], 1):
            if fidelity.level not in multi_fidelity_data:
                logger.warning(f"No data for fidelity level {fidelity.level}, skipping")
                continue

            X_fid, y_fid = multi_fidelity_data[fidelity.level]
            self._train_correction_model(X_fid, y_fid, fidelity.level, sorted_fidelities[:i])

            # Add data to fidelity level
            fidelity.add_data(X_fid, y_fid)

        self.is_trained = True
        logger.info("Hierarchical Multi-Fidelity Model training completed")

    def _train_base_model(self, X: np.ndarray, y: np.ndarray, fidelity_level: int):
        """Train the base (lowest fidelity) model"""
        kernel = self._create_kernel(X.shape[1])

        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            n_restarts_optimizer=10,
            normalize_y=True
        )

        model.fit(X, y)
        self.fidelity_models[fidelity_level] = model

        logger.info(f"Base model trained for fidelity {fidelity_level}")

    def _train_correction_model(self, X: np.ndarray, y: np.ndarray,
                              fidelity_level: int, lower_fidelities: List[FidelityLevel]):
        """
        Train correction model for a specific fidelity level.

        Args:
            X: Input data for current fidelity
            y: Output data for current fidelity
            fidelity_level: Current fidelity level
            lower_fidelities: List of lower fidelity levels
        """
        # Get predictions from lower fidelity hierarchy
        lower_predictions = self._get_hierarchical_predictions(X, lower_fidelities[-1].level)

        # Calculate correction (residual)
        if isinstance(lower_predictions, dict):
            # Extract mean predictions
            y_lower = np.array([pred['prediction'] for pred in lower_predictions.values()])
            if len(y_lower) > 1:
                y_lower = y_lower[0]  # Take first output if multi-output
        else:
            y_lower = lower_predictions

        correction = y - y_lower

        # Analyze correlation between lower and current fidelity
        correlation = np.corrcoef(y.flatten(), y_lower.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        self.fidelity_correlations[fidelity_level] = correlation

        # Choose model type based on correlation
        if abs(correlation) > self.correlation_threshold:
            # High correlation: use GP for smooth corrections
            if self.use_input_augmentation:
                # Augment inputs with lower fidelity predictions
                X_augmented = np.column_stack([X, y_lower.reshape(-1, 1)])
            else:
                X_augmented = X

            kernel = self._create_kernel(X_augmented.shape[1])

            correction_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=5,
                normalize_y=True
            )

        else:
            # Low correlation: use Random Forest for complex patterns
            correction_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            if self.use_input_augmentation:
                X_augmented = np.column_stack([X, y_lower.reshape(-1, 1)])
            else:
                X_augmented = X

        correction_model.fit(X_augmented, correction)
        self.fidelity_corrections[fidelity_level] = {
            'model': correction_model,
            'use_augmentation': self.use_input_augmentation,
            'correlation': correlation,
            'input_dim': X.shape[1]
        }

        logger.info(f"Correction model trained for fidelity {fidelity_level} "
                   f"(correlation: {correlation:.3f})")

    def _get_hierarchical_predictions(self, X: np.ndarray, target_fidelity: int) -> np.ndarray:
        """
        Get predictions by applying hierarchical corrections up to target fidelity.

        Args:
            X: Input points
            target_fidelity: Target fidelity level

        Returns:
            Predictions at target fidelity
        """
        # Start with base model prediction
        sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)
        base_fidelity = sorted_fidelities[0].level

        if base_fidelity not in self.fidelity_models:
            raise ValueError("Base model not trained")

        # Get base prediction
        base_pred = self.fidelity_models[base_fidelity].predict(X)
        current_pred = base_pred.copy()

        # Apply corrections hierarchically
        for fidelity in sorted_fidelities[1:]:
            if fidelity.level > target_fidelity:
                break

            if fidelity.level in self.fidelity_corrections:
                correction_info = self.fidelity_corrections[fidelity.level]
                correction_model = correction_info['model']

                if correction_info['use_augmentation']:
                    # Augment inputs with current predictions
                    X_augmented = np.column_stack([X, current_pred.reshape(-1, 1)])
                else:
                    X_augmented = X

                # Predict correction
                correction = correction_model.predict(X_augmented)
                current_pred += correction

        return current_pred

    def predict(self, X: np.ndarray, fidelity_level: int = None) -> Dict[str, Any]:
        """
        Make predictions with hierarchical multi-fidelity model.

        Args:
            X: Input points
            fidelity_level: Target fidelity level (None for highest)

        Returns:
            Predictions with uncertainty
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if fidelity_level is None:
            fidelity_level = self.get_highest_fidelity().level

        # Get hierarchical predictions
        mean_pred = self._get_hierarchical_predictions(X, fidelity_level)

        # Get uncertainty estimation
        uncertainty = self._estimate_hierarchical_uncertainty(X, fidelity_level)

        # Format results
        results = {}
        target_names = [f"output_{i}" for i in range(mean_pred.shape[1] if mean_pred.ndim > 1 else 1)]

        for i, target_name in enumerate(target_names):
            if mean_pred.ndim > 1:
                pred_val = float(mean_pred[0, i])
                std_val = float(uncertainty[0, i] if uncertainty.ndim > 1 else uncertainty[0])
            else:
                pred_val = float(mean_pred[0])
                std_val = float(uncertainty[0])

            results[target_name] = {
                'prediction': pred_val,
                'uncertainty': {
                    'standard_deviation': std_val,
                    'variance': std_val**2,
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ]
                },
                'fidelity_level': fidelity_level,
                'hierarchical_path': self._get_hierarchical_path(fidelity_level)
            }

        return results

    def _estimate_hierarchical_uncertainty(self, X: np.ndarray, fidelity_level: int) -> np.ndarray:
        """
        Estimate uncertainty by propagating through hierarchical structure.

        Args:
            X: Input points
            fidelity_level: Target fidelity level

        Returns:
            Uncertainty estimates
        """
        # Start with base model uncertainty
        sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)
        base_fidelity = sorted_fidelities[0].level

        base_model = self.fidelity_models[base_fidelity]
        if hasattr(base_model, 'predict') and hasattr(base_model, 'kernel_'):
            # GP model - get proper uncertainty
            _, base_std = base_model.predict(X, return_std=True)
            total_variance = base_std**2
        else:
            # Fallback uncertainty estimation
            total_variance = np.ones(X.shape[0]) * 0.1

        # Propagate uncertainty through corrections
        for fidelity in sorted_fidelities[1:]:
            if fidelity.level > fidelity_level:
                break

            if fidelity.level in self.fidelity_corrections:
                correction_info = self.fidelity_corrections[fidelity.level]
                correction_model = correction_info['model']

                if correction_info['use_augmentation']:
                    # This would require more sophisticated uncertainty propagation
                    # For now, add a simple correction uncertainty
                    correction_uncertainty = 0.05 * np.ones(X.shape[0])
                else:
                    if hasattr(correction_model, 'predict') and hasattr(correction_model, 'kernel_'):
                        _, correction_std = correction_model.predict(X, return_std=True)
                        correction_uncertainty = correction_std**2
                    else:
                        correction_uncertainty = 0.05 * np.ones(X.shape[0])

                # Add correction uncertainty
                total_variance += correction_uncertainty

        return np.sqrt(total_variance)

    def _get_hierarchical_path(self, fidelity_level: int) -> List[int]:
        """Get the hierarchical path to reach target fidelity level"""
        sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)
        path = []

        for fidelity in sorted_fidelities:
            path.append(fidelity.level)
            if fidelity.level >= fidelity_level:
                break

        return path

    def get_fidelity_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis of the hierarchical fidelity structure.

        Returns:
            Analysis of correlations, corrections, and model performance
        """
        analysis = {
            'fidelity_correlations': self.fidelity_correlations.copy(),
            'correction_models': {},
            'hierarchical_structure': self._get_hierarchical_structure(),
            'uncertainty_sources': self._analyze_uncertainty_sources()
        }

        # Analyze correction models
        for fid_level, correction_info in self.fidelity_corrections.items():
            model = correction_info['model']
            model_type = type(model).__name__

            analysis['correction_models'][fid_level] = {
                'model_type': model_type,
                'correlation': correction_info['correlation'],
                'uses_augmentation': correction_info['use_augmentation'],
                'input_dimension': correction_info['input_dim']
            }

            # Add model-specific information
            if hasattr(model, 'kernel_'):
                analysis['correction_models'][fid_level]['kernel'] = str(model.kernel_)
            elif hasattr(model, 'feature_importances_'):
                # For Random Forest
                analysis['correction_models'][fid_level]['feature_importance_std'] = float(
                    np.std(model.feature_importances_)
                )

        return analysis

    def _get_hierarchical_structure(self) -> Dict[str, Any]:
        """Analyze the hierarchical structure of the model"""
        sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)

        structure = {
            'base_fidelity': sorted_fidelities[0].level,
            'max_fidelity': sorted_fidelities[-1].level,
            'num_levels': len(sorted_fidelities),
            'correction_chain': [f.level for f in sorted_fidelities[1:]
                               if f.level in self.fidelity_corrections]
        }

        return structure

    def _analyze_uncertainty_sources(self) -> Dict[str, Any]:
        """Analyze different sources of uncertainty in the hierarchical model"""
        sources = {
            'base_model_uncertainty': 'GP posterior variance',
            'correction_uncertainties': [],
            'hierarchical_propagation': 'Uncertainty propagated through correction chain'
        }

        for fid_level in self.fidelity_corrections:
            correction_info = self.fidelity_corrections[fid_level]
            model_type = type(correction_info['model']).__name__

            sources['correction_uncertainties'].append({
                'fidelity_level': fid_level,
                'model_type': model_type,
                'correlation': correction_info['correlation']
            })

        return sources

    def update_with_new_data(self, X_new: np.ndarray, y_new: np.ndarray,
                           fidelity_level: int, retrain_corrections: bool = True):
        """
        Update the hierarchical model with new data.

        Args:
            X_new: New input data
            y_new: New output data
            fidelity_level: Fidelity level of new data
            retrain_corrections: Whether to retrain correction models
        """
        # Add data to fidelity level
        fidelity = self.get_fidelity_level(fidelity_level)
        fidelity.add_data(X_new, y_new)

        if fidelity_level == self.get_lowest_fidelity().level:
            # Retrain base model
            X_all, y_all = fidelity.get_all_data()
            self._train_base_model(X_all, y_all, fidelity_level)

            # Retrain all correction models if requested
            if retrain_corrections:
                sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)
                for i, fid in enumerate(sorted_fidelities[1:], 1):
                    if fid.n_samples > 0:
                        X_fid, y_fid = fid.get_all_data()
                        self._train_correction_model(X_fid, y_fid, fid.level, sorted_fidelities[:i])

        else:
            # Retrain correction model for this fidelity
            if retrain_corrections:
                sorted_fidelities = sorted(self.fidelity_levels, key=lambda x: x.level)
                fid_index = next(i for i, f in enumerate(sorted_fidelities) if f.level == fidelity_level)

                X_all, y_all = fidelity.get_all_data()
                self._train_correction_model(X_all, y_all, fidelity_level, sorted_fidelities[:fid_index])

        logger.info(f"Model updated with {len(X_new)} new samples at fidelity {fidelity_level}")