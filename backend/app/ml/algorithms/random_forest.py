import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ..base import SurrogateModelBase


class RandomForestSurrogate(SurrogateModelBase):
    """
    Random Forest surrogate model.

    Excellent for:
    - Medium to large datasets
    - Mixed categorical/numerical features
    - Feature importance analysis
    - Robust to outliers and noise

    Advantages:
    - Handles missing values well
    - Provides feature importance
    - Robust to overfitting
    - Fast training and prediction
    - No assumptions about data distribution

    Disadvantages:
    - Can overfit with very noisy data
    - Less smooth than GP or neural networks
    - Limited extrapolation capabilities
    - Large memory footprint for many trees
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)

    def _create_model(self) -> RandomForestRegressor:
        """Create Random Forest model"""
        return RandomForestRegressor(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            max_depth=self.hyperparameters.get('max_depth'),
            min_samples_split=self.hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
            max_features=self.hyperparameters.get('max_features', 'sqrt'),
            bootstrap=self.hyperparameters.get('bootstrap', True),
            oob_score=self.hyperparameters.get('oob_score', True),
            n_jobs=self.hyperparameters.get('n_jobs', -1),
            random_state=self.hyperparameters.get('random_state', 42)
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Random Forest model"""
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the Random Forest"""
        return self.model.predict(X)

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates using tree variance.

        Random Forest uncertainty is estimated using the variance
        of predictions across individual trees.
        """
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Calculate statistics
        mean_pred = np.mean(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)

        uncertainty = {}
        for i, target_name in enumerate(self.target_names):
            if len(self.target_names) == 1:
                std_val = float(std_pred[0])
                mean_val = float(mean_pred[0])
            else:
                std_val = float(std_pred[0, i])
                mean_val = float(mean_pred[0, i])

            uncertainty[target_name] = {
                'standard_deviation': std_val,
                'variance': std_val**2,
                'confidence_interval_95': [
                    mean_val - 1.96 * std_val,
                    mean_val + 1.96 * std_val
                ],
                'prediction_interval': self._get_prediction_interval(X, target_name)
            }

        return uncertainty

    def _get_prediction_interval(self, X: np.ndarray, target_name: str, confidence: float = 0.95) -> list:
        """
        Calculate prediction intervals using quantile regression trees.

        This provides more accurate intervals than Gaussian assumptions.
        """
        # Get all tree predictions
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Calculate quantiles
        alpha = 1 - confidence
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower_bound = np.quantile(tree_predictions, lower_quantile, axis=0)
        upper_bound = np.quantile(tree_predictions, upper_quantile, axis=0)

        if len(self.target_names) == 1:
            return [float(lower_bound[0]), float(upper_bound[0])]
        else:
            target_idx = self.target_names.index(target_name)
            return [float(lower_bound[0, target_idx]), float(upper_bound[0, target_idx])]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_oob_score(self) -> float:
        """Get out-of-bag score if available"""
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
            return float(self.model.oob_score_)
        return 0.0

    def get_tree_depths(self) -> Dict[str, Any]:
        """Get statistics about tree depths"""
        depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        return {
            'mean_depth': float(np.mean(depths)),
            'max_depth': int(np.max(depths)),
            'min_depth': int(np.min(depths)),
            'std_depth': float(np.std(depths))
        }

    def get_leaf_counts(self) -> Dict[str, Any]:
        """Get statistics about number of leaves"""
        leaf_counts = [tree.tree_.n_leaves for tree in self.model.estimators_]
        return {
            'mean_leaves': float(np.mean(leaf_counts)),
            'max_leaves': int(np.max(leaf_counts)),
            'min_leaves': int(np.min(leaf_counts)),
            'std_leaves': float(np.std(leaf_counts))
        }

    def partial_dependence(self, feature_idx: int, X_background: np.ndarray,
                          percentiles: tuple = (0.05, 0.95), grid_resolution: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate partial dependence for a specific feature.

        This shows the marginal effect of a feature on the prediction.
        """
        # Get feature range
        feature_min = np.percentile(X_background[:, feature_idx], percentiles[0] * 100)
        feature_max = np.percentile(X_background[:, feature_idx], percentiles[1] * 100)

        # Create grid for the feature
        feature_grid = np.linspace(feature_min, feature_max, grid_resolution)

        # Calculate partial dependence
        partial_deps = []
        for feature_val in feature_grid:
            # Create modified dataset with feature fixed at feature_val
            X_modified = X_background.copy()
            X_modified[:, feature_idx] = feature_val

            # Get predictions
            predictions = self._predict_model(X_modified)
            avg_prediction = np.mean(predictions, axis=0)
            partial_deps.append(avg_prediction)

        partial_deps = np.array(partial_deps)

        result = {
            'feature_values': feature_grid,
            'feature_name': self.feature_names[feature_idx]
        }

        for i, target_name in enumerate(self.target_names):
            if len(self.target_names) == 1:
                result[f'partial_dependence_{target_name}'] = partial_deps
            else:
                result[f'partial_dependence_{target_name}'] = partial_deps[:, i]

        return result

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
            'bootstrap': [True, False]
        }