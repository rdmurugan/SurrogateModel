from typing import Dict, Any, Type
from .base import SurrogateModelBase
from .algorithms import (
    GaussianProcessSurrogate,
    PolynomialChaosSurrogate,
    NeuralNetworkSurrogate,
    RandomForestSurrogate,
    SupportVectorSurrogate,
    RadialBasisSurrogate
)


class SurrogateModelFactory:
    """Factory class for creating surrogate models"""

    _algorithms: Dict[str, Type[SurrogateModelBase]] = {
        'gaussian_process': GaussianProcessSurrogate,
        'polynomial_chaos': PolynomialChaosSurrogate,
        'neural_network': NeuralNetworkSurrogate,
        'random_forest': RandomForestSurrogate,
        'support_vector': SupportVectorSurrogate,
        'radial_basis': RadialBasisSurrogate
    }

    @classmethod
    def create_model(cls, algorithm: str, **hyperparameters) -> SurrogateModelBase:
        """
        Create a surrogate model instance.

        Args:
            algorithm: Algorithm name (e.g., 'gaussian_process', 'neural_network')
            **hyperparameters: Algorithm-specific hyperparameters

        Returns:
            SurrogateModelBase: Configured surrogate model instance

        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm}' not supported. Available: {available}")

        model_class = cls._algorithms[algorithm]
        return model_class(**hyperparameters)

    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available algorithms.

        Returns:
            Dict containing algorithm names, descriptions, and default hyperparameters
        """
        algorithms_info = {}

        for name, model_class in cls._algorithms.items():
            # Get algorithm metadata
            doc = model_class.__doc__ or ""
            description = doc.split('\n')[0] if doc else f"{name} surrogate model"

            # Get default hyperparameters if available
            default_hyperparams = {}
            if hasattr(model_class, 'get_default_hyperparameters'):
                default_hyperparams = model_class.get_default_hyperparameters()

            # Extract key characteristics from docstring
            characteristics = cls._extract_characteristics(doc)

            algorithms_info[name] = {
                'name': name,
                'class_name': model_class.__name__,
                'description': description.strip(),
                'default_hyperparameters': default_hyperparams,
                'characteristics': characteristics
            }

        return algorithms_info

    @classmethod
    def _extract_characteristics(cls, docstring: str) -> Dict[str, list]:
        """Extract characteristics from algorithm docstring"""
        characteristics = {
            'excellent_for': [],
            'advantages': [],
            'disadvantages': []
        }

        if not docstring:
            return characteristics

        lines = docstring.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            if 'Excellent for:' in line:
                current_section = 'excellent_for'
            elif 'Advantages:' in line:
                current_section = 'advantages'
            elif 'Disadvantages:' in line:
                current_section = 'disadvantages'
            elif line.startswith('- ') and current_section:
                characteristics[current_section].append(line[2:])

        return characteristics

    @classmethod
    def recommend_algorithm(cls, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend the best algorithm based on dataset characteristics.

        Args:
            dataset_info: Dictionary with keys like 'n_samples', 'n_features', 'data_type'

        Returns:
            Dictionary with recommended algorithm and reasoning
        """
        n_samples = dataset_info.get('n_samples', 0)
        n_features = dataset_info.get('n_features', 0)
        data_type = dataset_info.get('data_type', 'numerical')

        recommendations = []

        # Small datasets (< 100 samples)
        if n_samples < 100:
            recommendations.append({
                'algorithm': 'gaussian_process',
                'score': 0.9,
                'reason': 'Gaussian Process works well with small datasets and provides uncertainty quantification'
            })
            recommendations.append({
                'algorithm': 'radial_basis',
                'score': 0.8,
                'reason': 'RBF can provide exact interpolation for small datasets'
            })

        # Medium datasets (100-1000 samples)
        elif n_samples < 1000:
            recommendations.append({
                'algorithm': 'random_forest',
                'score': 0.9,
                'reason': 'Random Forest is robust and provides feature importance for medium datasets'
            })
            recommendations.append({
                'algorithm': 'gaussian_process',
                'score': 0.8,
                'reason': 'Still feasible for medium datasets with good uncertainty quantification'
            })
            recommendations.append({
                'algorithm': 'support_vector',
                'score': 0.7,
                'reason': 'SVR handles medium datasets well and is robust to outliers'
            })

        # Large datasets (> 1000 samples)
        else:
            recommendations.append({
                'algorithm': 'neural_network',
                'score': 0.9,
                'reason': 'Neural networks excel with large datasets and complex patterns'
            })
            recommendations.append({
                'algorithm': 'random_forest',
                'score': 0.8,
                'reason': 'Scalable and robust for large datasets'
            })

        # High-dimensional data
        if n_features > 10:
            for rec in recommendations:
                if rec['algorithm'] in ['neural_network', 'support_vector']:
                    rec['score'] += 0.1
                    rec['reason'] += ' (good for high-dimensional data)'

        # If polynomial relationship is suspected
        if dataset_info.get('relationship_type') == 'polynomial':
            recommendations.append({
                'algorithm': 'polynomial_chaos',
                'score': 0.8,
                'reason': 'Polynomial Chaos Expansion is ideal for polynomial relationships'
            })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'all_recommendations': recommendations[:3],  # Top 3
            'dataset_characteristics': {
                'size_category': cls._get_size_category(n_samples),
                'dimensionality': 'high' if n_features > 10 else 'low',
                'recommended_for_uncertainty_quantification': n_samples < 500
            }
        }

    @classmethod
    def _get_size_category(cls, n_samples: int) -> str:
        """Categorize dataset size"""
        if n_samples < 100:
            return 'small'
        elif n_samples < 1000:
            return 'medium'
        else:
            return 'large'

    @classmethod
    def get_algorithm_comparison(cls) -> Dict[str, Dict[str, str]]:
        """
        Get a comparison matrix of algorithms across different criteria.

        Returns:
            Dictionary with algorithms as keys and criteria ratings as values
        """
        # Rating scale: Excellent, Good, Fair, Poor
        comparison = {
            'gaussian_process': {
                'small_datasets': 'Excellent',
                'large_datasets': 'Poor',
                'uncertainty_quantification': 'Excellent',
                'interpretability': 'Good',
                'computational_speed': 'Fair',
                'extrapolation': 'Good',
                'noise_robustness': 'Good'
            },
            'polynomial_chaos': {
                'small_datasets': 'Good',
                'large_datasets': 'Fair',
                'uncertainty_quantification': 'Excellent',
                'interpretability': 'Excellent',
                'computational_speed': 'Excellent',
                'extrapolation': 'Fair',
                'noise_robustness': 'Fair'
            },
            'neural_network': {
                'small_datasets': 'Poor',
                'large_datasets': 'Excellent',
                'uncertainty_quantification': 'Fair',
                'interpretability': 'Poor',
                'computational_speed': 'Good',
                'extrapolation': 'Fair',
                'noise_robustness': 'Good'
            },
            'random_forest': {
                'small_datasets': 'Fair',
                'large_datasets': 'Good',
                'uncertainty_quantification': 'Good',
                'interpretability': 'Good',
                'computational_speed': 'Good',
                'extrapolation': 'Poor',
                'noise_robustness': 'Excellent'
            },
            'support_vector': {
                'small_datasets': 'Good',
                'large_datasets': 'Fair',
                'uncertainty_quantification': 'Poor',
                'interpretability': 'Poor',
                'computational_speed': 'Fair',
                'extrapolation': 'Fair',
                'noise_robustness': 'Excellent'
            },
            'radial_basis': {
                'small_datasets': 'Excellent',
                'large_datasets': 'Poor',
                'uncertainty_quantification': 'Fair',
                'interpretability': 'Good',
                'computational_speed': 'Good',
                'extrapolation': 'Poor',
                'noise_robustness': 'Fair'
            }
        }

        return comparison