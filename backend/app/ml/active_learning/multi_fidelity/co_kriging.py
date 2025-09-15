import numpy as np
from typing import Dict, Any, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from .multi_fidelity_model import MultiFidelityModel


class CoKrigingModel(MultiFidelityModel):
    """
    Co-Kriging multi-fidelity model.

    Co-Kriging extends Gaussian Process regression to handle multiple fidelities
    by modeling the correlations between different fidelity levels. It's particularly
    effective when there's a strong correlation between low and high fidelity data.

    The approach models:
    - Low fidelity: f_l(x) ~ GP(μ_l, k_l)
    - High fidelity: f_h(x) = ρ * f_l(x) + δ(x)
    where δ(x) ~ GP(μ_δ, k_δ) is the discrepancy function
    """

    def __init__(self, fidelity_levels, correlation_prior: float = 0.8):
        """
        Initialize Co-Kriging model.

        Args:
            fidelity_levels: List of fidelity level definitions
            correlation_prior: Prior belief about correlation between fidelities
        """
        super().__init__(fidelity_levels)
        self.correlation_prior = correlation_prior
        self.low_fidelity_gp = None
        self.discrepancy_gp = None
        self.correlation_factor = None

    def fit(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        Train the Co-Kriging model.

        Args:
            multi_fidelity_data: Dict mapping fidelity level to (X, y) data
        """
        # Separate low and high fidelity data
        low_fidelity = self.get_lowest_fidelity()
        high_fidelity = self.get_highest_fidelity()

        if low_fidelity.level not in multi_fidelity_data:
            raise ValueError(f"No data provided for low fidelity level {low_fidelity.level}")

        if high_fidelity.level not in multi_fidelity_data:
            raise ValueError(f"No data provided for high fidelity level {high_fidelity.level}")

        X_low, y_low = multi_fidelity_data[low_fidelity.level]
        X_high, y_high = multi_fidelity_data[high_fidelity.level]

        # Train low fidelity GP
        self._train_low_fidelity_gp(X_low, y_low)

        # Calculate discrepancy and train discrepancy GP
        self._train_discrepancy_gp(X_high, y_high)

        # Store data in fidelity levels
        low_fidelity.add_data(X_low, y_low)
        high_fidelity.add_data(X_high, y_high)

        self.is_trained = True

    def _train_low_fidelity_gp(self, X_low: np.ndarray, y_low: np.ndarray):
        """Train Gaussian Process on low fidelity data"""
        kernel = ConstantKernel(1.0) * RBF(1.0)

        self.low_fidelity_gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            n_restarts_optimizer=10,
            normalize_y=True
        )

        self.low_fidelity_gp.fit(X_low, y_low)

    def _train_discrepancy_gp(self, X_high: np.ndarray, y_high: np.ndarray):
        """Train Gaussian Process on discrepancy between fidelities"""
        # Get low fidelity predictions at high fidelity points
        y_low_at_high, _ = self.low_fidelity_gp.predict(X_high, return_std=True)

        # Estimate correlation factor
        correlation = np.corrcoef(y_high.flatten(), y_low_at_high.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = self.correlation_prior

        self.correlation_factor = correlation

        # Calculate scaled low fidelity predictions
        y_low_scaled = self.correlation_factor * y_low_at_high

        # Calculate discrepancy
        discrepancy = y_high - y_low_scaled

        # Train discrepancy GP
        kernel = ConstantKernel(1.0) * RBF(1.0)

        self.discrepancy_gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            n_restarts_optimizer=10,
            normalize_y=True
        )

        self.discrepancy_gp.fit(X_high, discrepancy)

    def predict(self, X: np.ndarray, fidelity_level: int = None) -> Dict[str, Any]:
        """
        Make predictions with Co-Kriging model.

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

        if fidelity_level == self.get_lowest_fidelity().level:
            # Low fidelity prediction
            return self._predict_low_fidelity(X)
        elif fidelity_level == self.get_highest_fidelity().level:
            # High fidelity prediction
            return self._predict_high_fidelity(X)
        else:
            # Intermediate fidelity (interpolation)
            return self._predict_intermediate_fidelity(X, fidelity_level)

    def _predict_low_fidelity(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict using low fidelity model only"""
        mean_pred, std_pred = self.low_fidelity_gp.predict(X, return_std=True)

        results = {}
        target_names = [f"output_{i}" for i in range(mean_pred.shape[1] if mean_pred.ndim > 1 else 1)]

        for i, target_name in enumerate(target_names):
            if mean_pred.ndim > 1:
                pred_val = float(mean_pred[0, i])
                std_val = float(std_pred[0, i] if std_pred.ndim > 1 else std_pred[0])
            else:
                pred_val = float(mean_pred[0])
                std_val = float(std_pred[0])

            results[target_name] = {
                'prediction': pred_val,
                'uncertainty': {
                    'standard_deviation': std_val,
                    'variance': std_val**2,
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ]
                }
            }

        return results

    def _predict_high_fidelity(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict using full Co-Kriging model (high fidelity)"""
        # Low fidelity prediction
        y_low_mean, y_low_std = self.low_fidelity_gp.predict(X, return_std=True)

        # Discrepancy prediction
        discrepancy_mean, discrepancy_std = self.discrepancy_gp.predict(X, return_std=True)

        # Combined prediction
        mean_pred = self.correlation_factor * y_low_mean + discrepancy_mean

        # Combined uncertainty (simplified - assumes independence)
        var_low = (self.correlation_factor * y_low_std)**2
        var_discrepancy = discrepancy_std**2
        std_pred = np.sqrt(var_low + var_discrepancy)

        results = {}
        target_names = [f"output_{i}" for i in range(mean_pred.shape[1] if mean_pred.ndim > 1 else 1)]

        for i, target_name in enumerate(target_names):
            if mean_pred.ndim > 1:
                pred_val = float(mean_pred[0, i])
                std_val = float(std_pred[0, i] if std_pred.ndim > 1 else std_pred[0])
            else:
                pred_val = float(mean_pred[0])
                std_val = float(std_pred[0])

            results[target_name] = {
                'prediction': pred_val,
                'uncertainty': {
                    'standard_deviation': std_val,
                    'variance': std_val**2,
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ],
                    'low_fidelity_contribution': float(self.correlation_factor * y_low_mean[0]),
                    'discrepancy_contribution': float(discrepancy_mean[0]),
                    'correlation_factor': float(self.correlation_factor)
                }
            }

        return results

    def _predict_intermediate_fidelity(self, X: np.ndarray, fidelity_level: int) -> Dict[str, Any]:
        """Predict at intermediate fidelity level using interpolation"""
        # Get predictions at low and high fidelity
        low_pred = self._predict_low_fidelity(X)
        high_pred = self._predict_high_fidelity(X)

        # Interpolation weight based on fidelity level
        low_fidelity = self.get_lowest_fidelity()
        high_fidelity = self.get_highest_fidelity()

        weight = (fidelity_level - low_fidelity.level) / (high_fidelity.level - low_fidelity.level)

        results = {}
        for target_name in low_pred.keys():
            low_val = low_pred[target_name]['prediction']
            high_val = high_pred[target_name]['prediction']
            low_std = low_pred[target_name]['uncertainty']['standard_deviation']
            high_std = high_pred[target_name]['uncertainty']['standard_deviation']

            # Interpolated prediction and uncertainty
            pred_val = (1 - weight) * low_val + weight * high_val
            std_val = (1 - weight) * low_std + weight * high_std

            results[target_name] = {
                'prediction': float(pred_val),
                'uncertainty': {
                    'standard_deviation': float(std_val),
                    'variance': float(std_val**2),
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ],
                    'interpolation_weight': float(weight)
                }
            }

        return results

    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Get analysis of correlation between fidelity levels"""
        if not self.is_trained:
            return {}

        # Get training data
        low_fidelity = self.get_lowest_fidelity()
        high_fidelity = self.get_highest_fidelity()

        X_low, y_low = low_fidelity.get_all_data()
        X_high, y_high = high_fidelity.get_all_data()

        analysis = {
            'estimated_correlation': float(self.correlation_factor),
            'prior_correlation': float(self.correlation_prior),
            'correlation_strength': 'strong' if abs(self.correlation_factor) > 0.8 else
                                  'moderate' if abs(self.correlation_factor) > 0.5 else 'weak'
        }

        # Additional statistics if data is available
        if len(X_high) > 0:
            # Predictions at high fidelity points
            y_low_pred, _ = self.low_fidelity_gp.predict(X_high, return_std=True)

            # Discrepancy statistics
            discrepancy = y_high - self.correlation_factor * y_low_pred
            analysis.update({
                'mean_discrepancy': float(np.mean(discrepancy)),
                'std_discrepancy': float(np.std(discrepancy)),
                'max_discrepancy': float(np.max(np.abs(discrepancy))),
                'discrepancy_to_signal_ratio': float(np.std(discrepancy) / np.std(y_high))
            })

        return analysis

    def adaptive_correlation_update(self, X_new: np.ndarray, y_new_low: np.ndarray,
                                  y_new_high: np.ndarray):
        """
        Update correlation factor with new data.

        Args:
            X_new: New input points
            y_new_low: New low fidelity observations
            y_new_high: New high fidelity observations
        """
        # Get current low fidelity predictions
        y_low_pred, _ = self.low_fidelity_gp.predict(X_new, return_std=True)

        # Calculate new correlation
        all_low = np.append(y_low_pred, y_new_low)
        all_high = np.append(y_low_pred, y_new_high)  # Should be y_high from existing data

        new_correlation = np.corrcoef(all_low, all_high)[0, 1]

        if not np.isnan(new_correlation):
            # Update correlation with exponential smoothing
            alpha = 0.1  # Learning rate
            self.correlation_factor = (1 - alpha) * self.correlation_factor + alpha * new_correlation

    def cross_validate_fidelities(self, cv_folds: int = 5) -> Dict[str, float]:
        """
        Cross-validate the multi-fidelity model.

        Returns:
            Cross-validation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for cross-validation")

        # Get high fidelity data for CV
        high_fidelity = self.get_highest_fidelity()
        X_high, y_high = high_fidelity.get_all_data()

        if len(X_high) < cv_folds:
            return {'error': 'Insufficient high fidelity data for cross-validation'}

        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = []

        for train_idx, test_idx in kfold.split(X_high):
            X_train, X_test = X_high[train_idx], X_high[test_idx]
            y_train, y_test = y_high[train_idx], y_high[test_idx]

            # Create temporary model for this fold
            temp_model = CoKrigingModel(self.fidelity_levels, self.correlation_prior)

            # Train on fold data
            low_fidelity = self.get_lowest_fidelity()
            X_low, y_low = low_fidelity.get_all_data()

            fold_data = {
                low_fidelity.level: (X_low, y_low),
                high_fidelity.level: (X_train, y_train)
            }

            temp_model.fit(fold_data)

            # Predict and evaluate
            predictions = temp_model.predict(X_test, high_fidelity.level)

            # Extract predictions (simplified for single output)
            y_pred = [pred['prediction'] for pred in predictions.values()]

            # Calculate R² score
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
            cv_scores.append(score)

        return {
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'cv_scores': [float(score) for score in cv_scores]
        }