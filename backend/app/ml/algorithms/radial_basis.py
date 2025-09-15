import numpy as np
from typing import Dict, Any, Callable
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from ..base import SurrogateModelBase


class RadialBasisSurrogate(SurrogateModelBase):
    """
    Radial Basis Function (RBF) surrogate model.

    Excellent for:
    - Scattered data interpolation
    - Engineering design problems
    - Smooth function approximation
    - Small to medium datasets

    Advantages:
    - Exact interpolation possible
    - Good for smooth functions
    - Simple mathematical formulation
    - Fast evaluation once trained

    Disadvantages:
    - Can be ill-conditioned for large datasets
    - Sensitive to basis function selection
    - May require regularization
    - Limited extrapolation capabilities
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'basis_function': 'gaussian',
            'epsilon': 1.0,
            'smoothing': 0.0,
            'polynomial_degree': -1,  # -1 for no polynomial, 0 for constant, 1 for linear
            'center_selection': 'data',  # 'data', 'kmeans', 'random'
            'n_centers': None,  # If None, use all data points
            'regularization': 1e-10
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)
        self.centers = None
        self.weights = None
        self.polynomial_weights = None

    def _create_model(self) -> None:
        """RBF doesn't use sklearn models, so this returns None"""
        return None

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the RBF model"""
        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if y.ndim > 1 else 1

        # Select centers
        self.centers = self._select_centers(X)
        n_centers = len(self.centers)

        # Build RBF matrix
        rbf_matrix = self._build_rbf_matrix(X, self.centers)

        # Add polynomial terms if specified
        poly_degree = self.hyperparameters.get('polynomial_degree', -1)
        if poly_degree >= 0:
            poly_matrix = self._build_polynomial_matrix(X, poly_degree)
            # Augment the system
            n_poly_terms = poly_matrix.shape[1]

            # Build augmented matrix
            A = np.zeros((n_centers + n_poly_terms, n_centers + n_poly_terms))
            A[:n_centers, :n_centers] = rbf_matrix
            A[:n_centers, n_centers:] = poly_matrix
            A[n_centers:, :n_centers] = poly_matrix.T

            # Build augmented right-hand side
            b = np.zeros((n_centers + n_poly_terms, n_outputs))
            b[:n_centers] = y

            # Solve the system
            solution = solve(A, b)
            self.weights = solution[:n_centers]
            self.polynomial_weights = solution[n_centers:]

        else:
            # Add regularization
            regularization = self.hyperparameters.get('regularization', 1e-10)
            rbf_matrix += regularization * np.eye(n_centers)

            # Solve for weights
            self.weights = solve(rbf_matrix, y)
            self.polynomial_weights = None

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using RBF interpolation"""
        if self.centers is None or self.weights is None:
            raise ValueError("Model must be trained before making predictions")

        # Calculate RBF contributions
        rbf_values = self._evaluate_rbf(X, self.centers)
        predictions = rbf_values @ self.weights

        # Add polynomial contributions if present
        if self.polynomial_weights is not None:
            poly_degree = self.hyperparameters.get('polynomial_degree', -1)
            poly_values = self._build_polynomial_matrix(X, poly_degree)
            predictions += poly_values @ self.polynomial_weights

        return predictions

    def _select_centers(self, X: np.ndarray) -> np.ndarray:
        """Select RBF centers based on the specified method"""
        center_selection = self.hyperparameters.get('center_selection', 'data')
        n_centers = self.hyperparameters.get('n_centers')

        if center_selection == 'data':
            # Use all data points as centers
            return X.copy()

        elif center_selection == 'kmeans':
            # Use k-means clustering to select centers
            if n_centers is None:
                n_centers = min(len(X) // 2, 100)  # Default heuristic

            if n_centers >= len(X):
                return X.copy()

            kmeans = KMeans(n_clusters=n_centers, random_state=42)
            kmeans.fit(X)
            return kmeans.cluster_centers_

        elif center_selection == 'random':
            # Randomly select centers from data
            if n_centers is None:
                n_centers = min(len(X) // 2, 100)

            if n_centers >= len(X):
                return X.copy()

            indices = np.random.choice(len(X), size=n_centers, replace=False)
            return X[indices]

        else:
            raise ValueError(f"Unknown center selection method: {center_selection}")

    def _build_rbf_matrix(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Build the RBF interpolation matrix"""
        # Calculate pairwise distances
        distances = cdist(X, centers)

        # Apply basis function
        basis_function = self.hyperparameters.get('basis_function', 'gaussian')
        epsilon = self.hyperparameters.get('epsilon', 1.0)

        rbf_function = self._get_rbf_function(basis_function, epsilon)
        rbf_matrix = rbf_function(distances)

        return rbf_matrix

    def _evaluate_rbf(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Evaluate RBF functions at given points"""
        return self._build_rbf_matrix(X, centers)

    def _build_polynomial_matrix(self, X: np.ndarray, degree: int) -> np.ndarray:
        """Build polynomial basis matrix"""
        n_samples, n_features = X.shape

        if degree == 0:
            # Constant term only
            return np.ones((n_samples, 1))

        elif degree == 1:
            # Constant + linear terms
            poly_matrix = np.ones((n_samples, 1 + n_features))
            poly_matrix[:, 1:] = X
            return poly_matrix

        else:
            # Higher degree polynomials (simplified)
            from sklearn.preprocessing import PolynomialFeatures
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            return poly_features.fit_transform(X)

    def _get_rbf_function(self, function_name: str, epsilon: float) -> Callable:
        """Get the RBF function"""
        if function_name == 'gaussian':
            return lambda r: np.exp(-(epsilon * r)**2)

        elif function_name == 'multiquadric':
            return lambda r: np.sqrt(1 + (epsilon * r)**2)

        elif function_name == 'inverse_multiquadric':
            return lambda r: 1.0 / np.sqrt(1 + (epsilon * r)**2)

        elif function_name == 'linear':
            return lambda r: r

        elif function_name == 'cubic':
            return lambda r: r**3

        elif function_name == 'quintic':
            return lambda r: r**5

        elif function_name == 'thin_plate_spline':
            return lambda r: np.where(r == 0, 0, r**2 * np.log(r))

        else:
            raise ValueError(f"Unknown RBF function: {function_name}")

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates for RBF.

        RBF uncertainty is estimated using the condition number
        and distance to nearest centers.
        """
        predictions = self._predict_model(X)

        uncertainty = {}
        for i, target_name in enumerate(self.target_names):
            pred_val = float(predictions[0, i]) if predictions.ndim > 1 else float(predictions[0])

            # Distance-based uncertainty
            distances_to_centers = cdist(X, self.centers)
            min_distance = np.min(distances_to_centers[0])
            avg_distance = np.mean(distances_to_centers[0])

            # Condition number of RBF matrix (indicator of numerical stability)
            rbf_matrix = self._build_rbf_matrix(self.centers, self.centers)
            condition_number = np.linalg.cond(rbf_matrix)

            # Uncertainty increases with distance and condition number
            distance_factor = min_distance / (avg_distance + 1e-10)
            condition_factor = np.log10(condition_number + 1) / 10  # Normalize

            uncertainty_factor = (distance_factor + condition_factor) / 2
            std_estimate = abs(pred_val) * uncertainty_factor * 0.1

            uncertainty[target_name] = {
                'standard_deviation': std_estimate,
                'variance': std_estimate**2,
                'confidence_interval_95': [
                    pred_val - 1.96 * std_estimate,
                    pred_val + 1.96 * std_estimate
                ],
                'min_distance_to_center': float(min_distance),
                'condition_number': float(condition_number),
                'uncertainty_factor': float(uncertainty_factor)
            }

        return uncertainty

    def get_rbf_info(self) -> Dict[str, Any]:
        """Get information about the RBF model"""
        if self.centers is None:
            return {}

        rbf_matrix = self._build_rbf_matrix(self.centers, self.centers)

        return {
            'n_centers': len(self.centers),
            'basis_function': self.hyperparameters.get('basis_function'),
            'epsilon': self.hyperparameters.get('epsilon'),
            'condition_number': float(np.linalg.cond(rbf_matrix)),
            'matrix_rank': int(np.linalg.matrix_rank(rbf_matrix)),
            'has_polynomial': self.polynomial_weights is not None,
            'polynomial_degree': self.hyperparameters.get('polynomial_degree', -1)
        }

    def evaluate_at_centers(self) -> np.ndarray:
        """Evaluate the model at the center points (should be exact for interpolation)"""
        if self.centers is None:
            raise ValueError("Model not trained")
        return self._predict_model(self.centers)

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'basis_function': ['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline'],
            'epsilon': [0.1, 0.5, 1.0, 2.0, 5.0],
            'center_selection': ['data', 'kmeans'],
            'polynomial_degree': [-1, 0, 1],
            'regularization': [1e-12, 1e-10, 1e-8, 1e-6]
        }