import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.special import factorial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement
from ..base import SurrogateModelBase


class PolynomialChaosSurrogate(SurrogateModelBase):
    """
    Polynomial Chaos Expansion (PCE) surrogate model.

    Excellent for:
    - Problems with well-defined input distributions
    - Uncertainty quantification
    - Global sensitivity analysis
    - Analytical derivatives

    Advantages:
    - Fast evaluation once trained
    - Provides analytical expressions
    - Excellent for uncertainty propagation
    - Good for sensitivity analysis

    Disadvantages:
    - Curse of dimensionality for high-order polynomials
    - Assumes specific input distributions
    - May overfit with insufficient data
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'polynomial_order': 3,
            'distribution_type': 'uniform',  # uniform, normal
            'interaction_only': False,
            'include_bias': True,
            'sparse_regression': False,
            'alpha': 0.01  # For ridge regression if sparse_regression=True
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)
        self.polynomial_features = None
        self.polynomial_coefficients = None

    def _create_model(self) -> LinearRegression:
        """Create polynomial chaos expansion model"""
        polynomial_order = self.hyperparameters.get('polynomial_order', 3)
        interaction_only = self.hyperparameters.get('interaction_only', False)
        include_bias = self.hyperparameters.get('include_bias', True)

        self.polynomial_features = PolynomialFeatures(
            degree=polynomial_order,
            interaction_only=interaction_only,
            include_bias=include_bias
        )

        if self.hyperparameters.get('sparse_regression', False):
            from sklearn.linear_model import Ridge
            return Ridge(alpha=self.hyperparameters.get('alpha', 0.01))
        else:
            return LinearRegression()

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the polynomial chaos expansion"""
        # Transform input features to polynomial basis
        X_poly = self.polynomial_features.fit_transform(X)

        # Fit linear regression on polynomial features
        self.model.fit(X_poly, y)

        # Store coefficients for analysis
        self.polynomial_coefficients = self.model.coef_

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using polynomial expansion"""
        X_poly = self.polynomial_features.transform(X)
        return self.model.predict(X_poly)

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates using polynomial variance decomposition

        Note: This assumes uniform input distributions for simplicity.
        For accurate UQ, proper orthogonal polynomials should be used.
        """
        # For polynomial chaos, uncertainty comes from input uncertainty
        # This is a simplified implementation - proper PCE UQ requires
        # orthogonal polynomials and known input distributions

        predictions = self._predict_model(X)

        # Approximate uncertainty using coefficient magnitudes
        # This is a heuristic approach
        X_poly = self.polynomial_features.transform(X)

        uncertainty = {}
        for i, target_name in enumerate(self.target_names):
            if len(self.target_names) == 1:
                coeff_var = np.var(self.polynomial_coefficients)
                pred_std = np.sqrt(coeff_var * np.sum(X_poly**2, axis=1))
                std_val = float(pred_std[0])
            else:
                coeff_var = np.var(self.polynomial_coefficients[:, i])
                pred_std = np.sqrt(coeff_var * np.sum(X_poly**2, axis=1))
                std_val = float(pred_std[0])

            pred_val = float(predictions[0, i]) if predictions.ndim > 1 else float(predictions[0])

            uncertainty[target_name] = {
                'standard_deviation': std_val,
                'variance': std_val**2,
                'confidence_interval_95': [
                    pred_val - 1.96 * std_val,
                    pred_val + 1.96 * std_val
                ]
            }

        return uncertainty

    def get_sobol_indices(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate Sobol sensitivity indices using polynomial coefficients.

        This provides global sensitivity analysis capabilities.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to calculate Sobol indices")

        sobol_indices = {}

        for output_idx, target_name in enumerate(self.target_names):
            # Get coefficients for this output
            if len(self.target_names) == 1:
                coeffs = self.polynomial_coefficients.flatten()
            else:
                coeffs = self.polynomial_coefficients[:, output_idx]

            # Get polynomial feature names/powers
            feature_names = self.polynomial_features.get_feature_names_out(self.feature_names)

            # Calculate first-order Sobol indices
            first_order = {}
            total_variance = np.sum(coeffs[1:]**2)  # Exclude constant term

            for i, feature_name in enumerate(self.feature_names):
                # Find terms that only involve this feature
                first_order_terms = []
                for j, poly_name in enumerate(feature_names):
                    if feature_name in poly_name and len(poly_name.split()) == 1:
                        first_order_terms.append(j)

                first_order_variance = np.sum(coeffs[first_order_terms]**2)
                first_order[feature_name] = float(first_order_variance / total_variance) if total_variance > 0 else 0.0

            sobol_indices[target_name] = {
                'first_order': first_order,
                'total_variance': float(total_variance)
            }

        return sobol_indices

    def get_polynomial_expression(self, output_name: str = None) -> str:
        """
        Get the analytical polynomial expression.

        Useful for understanding the model structure and manual analysis.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get polynomial expression")

        if output_name is None:
            output_name = self.target_names[0]

        output_idx = self.target_names.index(output_name)

        if len(self.target_names) == 1:
            coeffs = self.polynomial_coefficients.flatten()
        else:
            coeffs = self.polynomial_coefficients[:, output_idx]

        feature_names = self.polynomial_features.get_feature_names_out(self.feature_names)

        expression = f"{output_name} = "
        terms = []

        for i, (coeff, feature) in enumerate(zip(coeffs, feature_names)):
            if abs(coeff) > 1e-10:  # Skip very small coefficients
                if feature == "1":  # Constant term
                    terms.append(f"{coeff:.6f}")
                else:
                    terms.append(f"{coeff:.6f} * {feature}")

        expression += " + ".join(terms)
        return expression

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on polynomial coefficients"""
        if not self.is_trained:
            return {}

        # Calculate importance as sum of squared coefficients for each input feature
        feature_names = self.polynomial_features.get_feature_names_out(self.feature_names)

        if len(self.target_names) == 1:
            coeffs = self.polynomial_coefficients.flatten()
        else:
            # Average importance across all outputs
            coeffs = np.mean(np.abs(self.polynomial_coefficients), axis=1)

        importance = {}
        for feature in self.feature_names:
            # Sum coefficients for all terms involving this feature
            feature_importance = 0.0
            for i, poly_feature in enumerate(feature_names):
                if feature in poly_feature and i < len(coeffs):
                    feature_importance += abs(coeffs[i])**2

            importance[feature] = float(feature_importance)

        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}

        return importance

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'polynomial_order': [2, 3, 4, 5],
            'interaction_only': [False, True],
            'sparse_regression': [False, True],
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }