from typing import Dict, Any, Type
from .base import AcquisitionFunction
from .expected_improvement import ExpectedImprovement
from .probability_improvement import ProbabilityOfImprovement
from .upper_confidence_bound import UpperConfidenceBound
from .knowledge_gradient import KnowledgeGradient


class AcquisitionFunctionFactory:
    """Factory for creating acquisition functions"""

    _functions: Dict[str, Type[AcquisitionFunction]] = {
        'expected_improvement': ExpectedImprovement,
        'ei': ExpectedImprovement,  # Alias
        'probability_improvement': ProbabilityOfImprovement,
        'pi': ProbabilityOfImprovement,  # Alias
        'upper_confidence_bound': UpperConfidenceBound,
        'ucb': UpperConfidenceBound,  # Alias
        'knowledge_gradient': KnowledgeGradient,
        'kg': KnowledgeGradient  # Alias
    }

    @classmethod
    def create(cls, function_name: str, **kwargs) -> AcquisitionFunction:
        """
        Create an acquisition function instance.

        Args:
            function_name: Name of the acquisition function
            **kwargs: Function-specific hyperparameters

        Returns:
            AcquisitionFunction instance

        Raises:
            ValueError: If function_name is not supported
        """
        if function_name not in cls._functions:
            available = list(cls._functions.keys())
            raise ValueError(f"Acquisition function '{function_name}' not supported. Available: {available}")

        function_class = cls._functions[function_name]
        return function_class(**kwargs)

    @classmethod
    def get_available_functions(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available acquisition functions.

        Returns:
            Dictionary with function information
        """
        functions_info = {}

        # Map aliases to their primary names
        primary_names = {
            'expected_improvement': ExpectedImprovement,
            'probability_improvement': ProbabilityOfImprovement,
            'upper_confidence_bound': UpperConfidenceBound,
            'knowledge_gradient': KnowledgeGradient
        }

        for name, function_class in primary_names.items():
            doc = function_class.__doc__ or ""
            description = doc.split('\n')[0] if doc else f"{name} acquisition function"

            # Get default hyperparameters
            default_hyperparams = {}
            if hasattr(function_class, 'get_default_hyperparameters'):
                default_hyperparams = function_class.get_default_hyperparameters()

            # Extract characteristics
            characteristics = cls._extract_characteristics(doc)

            # Get aliases
            aliases = [alias for alias, cls_type in cls._functions.items()
                      if cls_type == function_class and alias != name]

            functions_info[name] = {
                'name': name,
                'aliases': aliases,
                'class_name': function_class.__name__,
                'description': description.strip(),
                'default_hyperparameters': default_hyperparams,
                'characteristics': characteristics,
                'use_cases': cls._get_use_cases(name)
            }

        return functions_info

    @classmethod
    def _extract_characteristics(cls, docstring: str) -> Dict[str, str]:
        """Extract key characteristics from docstring"""
        characteristics = {
            'exploration_exploitation': 'balanced',
            'computational_cost': 'medium',
            'gradient_available': 'finite_difference'
        }

        if not docstring:
            return characteristics

        doc_lower = docstring.lower()

        # Determine exploration/exploitation behavior
        if 'exploit' in doc_lower and 'explore' not in doc_lower:
            characteristics['exploration_exploitation'] = 'exploitative'
        elif 'explore' in doc_lower and 'exploit' not in doc_lower:
            characteristics['exploration_exploitation'] = 'exploratory'

        # Determine computational cost
        if 'monte carlo' in doc_lower or 'fantasy' in doc_lower:
            characteristics['computational_cost'] = 'high'
        elif 'simple' in doc_lower or 'fast' in doc_lower:
            characteristics['computational_cost'] = 'low'

        return characteristics

    @classmethod
    def _get_use_cases(cls, function_name: str) -> Dict[str, str]:
        """Get specific use cases for each acquisition function"""
        use_cases = {
            'expected_improvement': {
                'best_for': 'General-purpose Bayesian optimization',
                'when_to_use': 'Balanced exploration/exploitation needed',
                'engineering_applications': 'Design optimization, parameter tuning',
                'strengths': 'Well-established, robust performance',
                'limitations': 'Can be conservative in high dimensions'
            },
            'probability_improvement': {
                'best_for': 'Quick convergence to local optima',
                'when_to_use': 'When exploitation is more important than exploration',
                'engineering_applications': 'Fine-tuning around known good designs',
                'strengths': 'Simple, fast computation',
                'limitations': 'Overly exploitative, poor global search'
            },
            'upper_confidence_bound': {
                'best_for': 'Regret-bounded optimization',
                'when_to_use': 'When you need theoretical guarantees',
                'engineering_applications': 'Safety-critical optimization, robust design',
                'strengths': 'Theoretical guarantees, tunable exploration',
                'limitations': 'Can be overly exploratory initially'
            },
            'knowledge_gradient': {
                'best_for': 'Information-theoretic optimization',
                'when_to_use': 'When sample efficiency is critical',
                'engineering_applications': 'Expensive simulations, limited budget',
                'strengths': 'Excellent sample efficiency',
                'limitations': 'Computationally expensive'
            }
        }

        return use_cases.get(function_name, {})

    @classmethod
    def recommend_function(cls, problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend the best acquisition function based on problem characteristics.

        Args:
            problem_characteristics: Dict with keys like 'budget', 'dimensions', 'objective'

        Returns:
            Recommendation with reasoning
        """
        budget = problem_characteristics.get('budget', 'medium')
        dimensions = problem_characteristics.get('dimensions', 5)
        objective = problem_characteristics.get('objective', 'optimization')
        noise_level = problem_characteristics.get('noise_level', 'low')
        multi_modal = problem_characteristics.get('multi_modal', False)

        recommendations = []

        # Budget considerations
        if budget == 'low':
            recommendations.append({
                'function': 'knowledge_gradient',
                'score': 0.9,
                'reason': 'Excellent sample efficiency for limited budgets'
            })
        elif budget == 'high':
            recommendations.append({
                'function': 'upper_confidence_bound',
                'score': 0.8,
                'reason': 'Good exploration for high-budget scenarios'
            })

        # Dimensionality considerations
        if dimensions > 20:
            recommendations.append({
                'function': 'upper_confidence_bound',
                'score': 0.8,
                'reason': 'Handles high dimensions well'
            })
        else:
            recommendations.append({
                'function': 'expected_improvement',
                'score': 0.9,
                'reason': 'Excellent for moderate dimensions'
            })

        # Multi-modal considerations
        if multi_modal:
            recommendations.append({
                'function': 'upper_confidence_bound',
                'score': 0.8,
                'reason': 'Good global exploration for multi-modal functions'
            })

        # Noise considerations
        if noise_level == 'high':
            recommendations.append({
                'function': 'upper_confidence_bound',
                'score': 0.8,
                'reason': 'Robust to noisy observations'
            })

        # Default recommendation
        if not recommendations:
            recommendations.append({
                'function': 'expected_improvement',
                'score': 0.8,
                'reason': 'Reliable general-purpose choice'
            })

        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        return {
            'primary_recommendation': recommendations[0],
            'all_recommendations': recommendations[:3],
            'problem_analysis': {
                'complexity': 'high' if dimensions > 10 or multi_modal else 'low',
                'sample_efficiency_critical': budget == 'low',
                'exploration_needed': multi_modal or noise_level == 'high'
            }
        }

    @classmethod
    def get_function_comparison(cls) -> Dict[str, Dict[str, str]]:
        """
        Get a comparison matrix of acquisition functions.

        Returns:
            Comparison matrix with ratings
        """
        comparison = {
            'expected_improvement': {
                'exploration_exploitation': 'Balanced',
                'sample_efficiency': 'Good',
                'computational_cost': 'Low',
                'theoretical_guarantees': 'None',
                'multi_modal_performance': 'Good',
                'high_dimensional_performance': 'Fair',
                'noise_robustness': 'Good'
            },
            'probability_improvement': {
                'exploration_exploitation': 'Exploitative',
                'sample_efficiency': 'Fair',
                'computational_cost': 'Very Low',
                'theoretical_guarantees': 'None',
                'multi_modal_performance': 'Poor',
                'high_dimensional_performance': 'Fair',
                'noise_robustness': 'Fair'
            },
            'upper_confidence_bound': {
                'exploration_exploitation': 'Exploratory',
                'sample_efficiency': 'Good',
                'computational_cost': 'Low',
                'theoretical_guarantees': 'Yes',
                'multi_modal_performance': 'Excellent',
                'high_dimensional_performance': 'Good',
                'noise_robustness': 'Excellent'
            },
            'knowledge_gradient': {
                'exploration_exploitation': 'Adaptive',
                'sample_efficiency': 'Excellent',
                'computational_cost': 'High',
                'theoretical_guarantees': 'Information-theoretic',
                'multi_modal_performance': 'Good',
                'high_dimensional_performance': 'Good',
                'noise_robustness': 'Good'
            }
        }

        return comparison