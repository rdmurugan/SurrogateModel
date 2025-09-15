import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
from .base_sampler import BaseSampler
from .batch_sampler import BatchActiveLearning
from .physics_informed_sampler import PhysicsInformedSampler


class AdaptiveSampler(BaseSampler):
    """
    Adaptive sampling strategy that dynamically adjusts its approach based on
    model performance, data characteristics, and sampling history.

    This meta-sampler orchestrates different sampling strategies and selects
    the most appropriate one based on the current state of the active learning
    process. It provides intelligent adaptation across:

    1. Strategy selection (batch, physics-informed, uncertainty-based)
    2. Parameter tuning based on performance feedback
    3. Multi-fidelity adaptation
    4. Budget-aware optimization
    5. Convergence detection and strategy switching
    """

    def __init__(self,
                 strategies: Dict[str, Any] = None,
                 adaptation_frequency: int = 5,
                 performance_window: int = 10,
                 budget_constraints: Dict[str, float] = None,
                 convergence_threshold: float = 1e-4):
        """
        Initialize adaptive sampler.

        Args:
            strategies: Dictionary of available sampling strategies
            adaptation_frequency: How often to reassess strategy (in iterations)
            performance_window: Window size for performance evaluation
            budget_constraints: Budget constraints for different operations
            convergence_threshold: Threshold for convergence detection
        """
        super().__init__()

        # Initialize available strategies
        self.strategies = self._initialize_strategies(strategies)
        self.current_strategy = "uncertainty_based"  # Default strategy

        # Adaptation parameters
        self.adaptation_frequency = adaptation_frequency
        self.performance_window = performance_window
        self.convergence_threshold = convergence_threshold

        # Budget management
        self.budget_constraints = budget_constraints or {}
        self.budget_used = 0.0
        self.budget_efficiency = {}

        # Performance tracking
        self.strategy_performance = {name: [] for name in self.strategies.keys()}
        self.convergence_history = []
        self.adaptation_decisions = []

        # State tracking
        self.iteration_count = 0
        self.last_adaptation_iteration = 0
        self.model_performance_trend = []

    def _initialize_strategies(self, strategies: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initialize available sampling strategies"""
        if strategies is None:
            strategies = {}

        default_strategies = {
            "uncertainty_based": {
                "type": "acquisition",
                "name": "upper_confidence_bound",
                "params": {"kappa": 2.0}
            },
            "batch_sampling": {
                "type": "batch",
                "sampler": BatchActiveLearning(),
                "params": {"strategy": "diversity_aware", "batch_size": 3}
            },
            "physics_informed": {
                "type": "physics",
                "sampler": PhysicsInformedSampler(),
                "params": {"physics_constraints": {}, "boundary_weights": {}}
            },
            "exploration": {
                "type": "acquisition",
                "name": "probability_improvement",
                "params": {"xi": 0.1}
            },
            "exploitation": {
                "type": "acquisition",
                "name": "expected_improvement",
                "params": {"xi": 0.01}
            }
        }

        # Merge with provided strategies
        for name, config in default_strategies.items():
            if name not in strategies:
                strategies[name] = config

        return strategies

    def sample(self,
               model,
               X_candidates: np.ndarray,
               n_samples: int = 1,
               acquisition_function=None,
               **kwargs) -> Dict[str, Any]:
        """
        Adaptively sample points using the most appropriate strategy.

        Args:
            model: Trained surrogate model
            X_candidates: Candidate sampling points
            n_samples: Number of points to sample
            acquisition_function: Base acquisition function
            **kwargs: Additional parameters

        Returns:
            Adaptive sampling results with strategy information
        """
        self.iteration_count += 1

        # Determine if adaptation is needed
        if self._should_adapt():
            self._adapt_strategy(model, X_candidates)

        # Execute current strategy
        strategy_result = self._execute_strategy(
            model, X_candidates, n_samples, acquisition_function, **kwargs
        )

        # Update performance tracking
        self._update_performance_tracking(strategy_result, model)

        # Prepare adaptive result
        result = {
            **strategy_result,
            'adaptive_info': {
                'current_strategy': self.current_strategy,
                'iteration': self.iteration_count,
                'adaptation_decisions': self.adaptation_decisions[-5:],  # Last 5 decisions
                'strategy_performance': self._get_strategy_performance_summary(),
                'budget_status': self._get_budget_status(),
                'convergence_indicators': self._get_convergence_indicators()
            }
        }

        # Update history
        self.update_history(result)

        return result

    def _should_adapt(self) -> bool:
        """Determine if strategy adaptation is needed"""
        # Adapt every N iterations
        if self.iteration_count - self.last_adaptation_iteration >= self.adaptation_frequency:
            return True

        # Adapt if performance is declining
        if len(self.strategy_performance[self.current_strategy]) >= 3:
            recent_performance = self.strategy_performance[self.current_strategy][-3:]
            if all(recent_performance[i] >= recent_performance[i+1] for i in range(len(recent_performance)-1)):
                return True

        # Adapt if convergence is detected
        if self._detect_convergence():
            return True

        return False

    def _adapt_strategy(self, model, X_candidates: np.ndarray):
        """Adapt the sampling strategy based on current state"""
        self.last_adaptation_iteration = self.iteration_count

        # Analyze current state
        state_analysis = self._analyze_current_state(model, X_candidates)

        # Select best strategy
        new_strategy = self._select_strategy(state_analysis)

        # Record adaptation decision
        decision = {
            'iteration': self.iteration_count,
            'previous_strategy': self.current_strategy,
            'new_strategy': new_strategy,
            'reason': state_analysis['adaptation_reason'],
            'state_analysis': state_analysis
        }

        self.adaptation_decisions.append(decision)
        self.current_strategy = new_strategy

    def _analyze_current_state(self, model, X_candidates: np.ndarray) -> Dict[str, Any]:
        """Analyze current state to inform strategy selection"""
        analysis = {}

        # Model performance analysis
        if hasattr(model, 'training_metrics'):
            metrics = model.training_metrics
            analysis['model_accuracy'] = metrics.get('r2_score', 0.0)
            analysis['model_uncertainty'] = metrics.get('mean_prediction_std', 1.0)
        else:
            analysis['model_accuracy'] = 0.5  # Default
            analysis['model_uncertainty'] = 1.0

        # Data analysis
        analysis['n_training_samples'] = self._get_training_sample_count(model)
        analysis['data_density'] = self._estimate_data_density(X_candidates)

        # Budget analysis
        analysis['budget_remaining'] = self._get_remaining_budget()
        analysis['budget_efficiency'] = self._calculate_budget_efficiency()

        # Convergence analysis
        analysis['convergence_trend'] = self._analyze_convergence_trend()
        analysis['exploration_completeness'] = self._estimate_exploration_completeness(X_candidates)

        # Strategy performance analysis
        analysis['strategy_rankings'] = self._rank_strategies()

        # Determine adaptation reason
        analysis['adaptation_reason'] = self._determine_adaptation_reason(analysis)

        return analysis

    def _select_strategy(self, state_analysis: Dict[str, Any]) -> str:
        """Select the most appropriate strategy based on state analysis"""
        scores = {}

        for strategy_name in self.strategies.keys():
            scores[strategy_name] = self._calculate_strategy_score(strategy_name, state_analysis)

        # Select strategy with highest score
        best_strategy = max(scores.keys(), key=lambda k: scores[k])

        return best_strategy

    def _calculate_strategy_score(self, strategy_name: str, state_analysis: Dict[str, Any]) -> float:
        """Calculate score for a strategy given current state"""
        score = 0.0

        # Base performance score
        if strategy_name in self.strategy_performance:
            recent_performance = self.strategy_performance[strategy_name][-5:]
            if recent_performance:
                score += np.mean(recent_performance) * 0.3

        # State-specific scoring
        if strategy_name == "uncertainty_based":
            # Good for exploration when model uncertainty is high
            score += state_analysis['model_uncertainty'] * 0.4
            score += (1 - state_analysis['exploration_completeness']) * 0.3

        elif strategy_name == "batch_sampling":
            # Good when budget allows parallel sampling
            if state_analysis['budget_remaining'] > 0.2:
                score += 0.5
            score += state_analysis['data_density'] * 0.3

        elif strategy_name == "physics_informed":
            # Good when model accuracy is reasonable and we want targeted sampling
            if state_analysis['model_accuracy'] > 0.3:
                score += 0.4
            score += state_analysis['budget_efficiency'] * 0.3

        elif strategy_name == "exploration":
            # Good early in the process
            if state_analysis['n_training_samples'] < 50:
                score += 0.6
            score += (1 - state_analysis['exploration_completeness']) * 0.4

        elif strategy_name == "exploitation":
            # Good when we have good model and want to refine
            score += state_analysis['model_accuracy'] * 0.5
            if state_analysis['convergence_trend'] < 0.1:  # Converging
                score += 0.3

        return score

    def _execute_strategy(self,
                         model,
                         X_candidates: np.ndarray,
                         n_samples: int,
                         acquisition_function,
                         **kwargs) -> Dict[str, Any]:
        """Execute the current sampling strategy"""
        strategy_config = self.strategies[self.current_strategy]
        strategy_type = strategy_config['type']

        if strategy_type == "acquisition":
            return self._execute_acquisition_strategy(
                model, X_candidates, n_samples, strategy_config, **kwargs
            )
        elif strategy_type == "batch":
            return self._execute_batch_strategy(
                model, X_candidates, n_samples, strategy_config, **kwargs
            )
        elif strategy_type == "physics":
            return self._execute_physics_strategy(
                model, X_candidates, n_samples, strategy_config, **kwargs
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _execute_acquisition_strategy(self,
                                    model,
                                    X_candidates: np.ndarray,
                                    n_samples: int,
                                    strategy_config: Dict[str, Any],
                                    **kwargs) -> Dict[str, Any]:
        """Execute acquisition function based strategy"""
        # Import acquisition functions
        from ..acquisition.factory import AcquisitionFunctionFactory

        acq_name = strategy_config['name']
        acq_params = strategy_config.get('params', {})

        # Create acquisition function
        acq_func = AcquisitionFunctionFactory.create(acq_name, **acq_params)

        # Set current best if needed
        if hasattr(model, 'training_data'):
            # Extract best value from training data
            _, y_train = model.training_data
            if len(y_train) > 0:
                current_best = np.max(y_train)
                acq_func.set_current_best(current_best)

        # Evaluate acquisition function
        acq_values = acq_func.evaluate(X_candidates, model)

        # Select top n_samples points
        if n_samples == 1:
            selected_indices = np.array([np.argmax(acq_values)])
        else:
            # Use diverse selection for multiple points
            selected_indices = self._diverse_selection(X_candidates, acq_values, n_samples)

        selected_points = X_candidates[selected_indices]

        return {
            'selected_points': selected_points,
            'selected_indices': selected_indices,
            'acquisition_scores': acq_values[selected_indices],
            'strategy_type': 'acquisition',
            'strategy_name': acq_name
        }

    def _execute_batch_strategy(self,
                              model,
                              X_candidates: np.ndarray,
                              n_samples: int,
                              strategy_config: Dict[str, Any],
                              **kwargs) -> Dict[str, Any]:
        """Execute batch sampling strategy"""
        sampler = strategy_config['sampler']
        params = strategy_config.get('params', {})

        # Update batch size if needed
        if 'batch_size' in params:
            params['batch_size'] = min(params['batch_size'], n_samples)

        result = sampler.sample(model, X_candidates, n_samples, **params)

        result.update({
            'strategy_type': 'batch',
            'strategy_name': 'batch_sampling'
        })

        return result

    def _execute_physics_strategy(self,
                                model,
                                X_candidates: np.ndarray,
                                n_samples: int,
                                strategy_config: Dict[str, Any],
                                **kwargs) -> Dict[str, Any]:
        """Execute physics-informed strategy"""
        sampler = strategy_config['sampler']
        params = strategy_config.get('params', {})

        result = sampler.sample(model, X_candidates, n_samples, **params)

        result.update({
            'strategy_type': 'physics',
            'strategy_name': 'physics_informed'
        })

        return result

    def _diverse_selection(self, X_candidates: np.ndarray, scores: np.ndarray, n_samples: int) -> np.ndarray:
        """Select diverse points with high acquisition scores"""
        if n_samples >= len(X_candidates):
            return np.arange(len(X_candidates))

        selected_indices = []

        for _ in range(n_samples):
            if len(selected_indices) == 0:
                # Select point with highest score
                idx = np.argmax(scores)
            else:
                # Balance score and diversity
                remaining_indices = np.setdiff1d(np.arange(len(X_candidates)), selected_indices)

                if len(remaining_indices) == 0:
                    break

                # Calculate diversity weights
                selected_points = X_candidates[selected_indices]
                remaining_points = X_candidates[remaining_indices]

                distances = cdist(remaining_points, selected_points)
                min_distances = np.min(distances, axis=1)

                # Combine score and diversity
                combined_scores = scores[remaining_indices] * (1 + min_distances)
                best_relative_idx = np.argmax(combined_scores)
                idx = remaining_indices[best_relative_idx]

            selected_indices.append(idx)

        return np.array(selected_indices)

    def _update_performance_tracking(self, result: Dict[str, Any], model):
        """Update performance tracking for the current strategy"""
        # Calculate performance metric (could be acquisition value, model improvement, etc.)
        if 'acquisition_scores' in result:
            performance = np.mean(result['acquisition_scores'])
        else:
            performance = 1.0  # Default performance

        self.strategy_performance[self.current_strategy].append(performance)

        # Limit history size
        max_history = self.performance_window * 2
        if len(self.strategy_performance[self.current_strategy]) > max_history:
            self.strategy_performance[self.current_strategy] = \
                self.strategy_performance[self.current_strategy][-max_history:]

    def _get_training_sample_count(self, model) -> int:
        """Get number of training samples from model"""
        if hasattr(model, 'training_data'):
            X_train, _ = model.training_data
            return len(X_train)
        elif hasattr(model, 'n_samples_'):
            return model.n_samples_
        else:
            return 10  # Default estimate

    def _estimate_data_density(self, X_candidates: np.ndarray) -> float:
        """Estimate data density in the candidate space"""
        if len(X_candidates) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = cdist(X_candidates, X_candidates)
        np.fill_diagonal(distances, np.inf)

        # Average minimum distance as density indicator
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)

        # Convert to density score (0-1)
        density_score = 1.0 / (1.0 + avg_min_distance)

        return density_score

    def _get_remaining_budget(self) -> float:
        """Get remaining budget ratio"""
        total_budget = self.budget_constraints.get('total', 1000.0)
        remaining_ratio = (total_budget - self.budget_used) / total_budget
        return max(0.0, remaining_ratio)

    def _calculate_budget_efficiency(self) -> float:
        """Calculate budget efficiency from history"""
        if not self.history:
            return 0.5

        # Simplified efficiency calculation
        recent_results = self.history[-5:]
        total_cost = sum(len(result['sample_result'].get('selected_points', [])) for result in recent_results)

        if total_cost == 0:
            return 0.5

        # Efficiency based on acquisition scores
        total_value = sum(
            np.sum(result['sample_result'].get('acquisition_scores', [0]))
            for result in recent_results
        )

        efficiency = total_value / total_cost if total_cost > 0 else 0.5

        return min(1.0, efficiency)

    def _analyze_convergence_trend(self) -> float:
        """Analyze convergence trend from recent iterations"""
        if len(self.convergence_history) < 3:
            return 1.0  # Not converging yet

        recent_convergence = self.convergence_history[-5:]
        if len(recent_convergence) < 2:
            return 1.0

        # Calculate trend (positive = diverging, negative = converging)
        trend = np.polyfit(range(len(recent_convergence)), recent_convergence, 1)[0]

        return max(0.0, -trend)  # Convert to convergence score

    def _estimate_exploration_completeness(self, X_candidates: np.ndarray) -> float:
        """Estimate how completely the space has been explored"""
        # Simplified estimation based on candidate density
        if len(X_candidates) == 0:
            return 1.0

        # Use data density as proxy for exploration completeness
        density = self._estimate_data_density(X_candidates)

        # Higher density suggests more complete exploration
        completeness = min(1.0, density * 2.0)

        return completeness

    def _rank_strategies(self) -> Dict[str, float]:
        """Rank strategies based on recent performance"""
        rankings = {}

        for strategy_name, performance_history in self.strategy_performance.items():
            if performance_history:
                # Recent performance weighted more heavily
                weights = np.exp(np.linspace(-1, 0, len(performance_history)))
                weighted_performance = np.average(performance_history, weights=weights)
                rankings[strategy_name] = weighted_performance
            else:
                rankings[strategy_name] = 0.0

        return rankings

    def _determine_adaptation_reason(self, analysis: Dict[str, Any]) -> str:
        """Determine the primary reason for strategy adaptation"""
        if analysis['model_accuracy'] < 0.3:
            return "low_model_accuracy"
        elif analysis['budget_remaining'] < 0.2:
            return "low_budget"
        elif analysis['exploration_completeness'] < 0.5:
            return "incomplete_exploration"
        elif analysis['convergence_trend'] < 0.1:
            return "converging"
        else:
            return "performance_optimization"

    def _detect_convergence(self) -> bool:
        """Detect if the active learning process is converging"""
        if len(self.convergence_history) < 5:
            return False

        recent_values = self.convergence_history[-5:]
        # Check if recent improvements are below threshold
        max_improvement = max(recent_values) - min(recent_values)

        return max_improvement < self.convergence_threshold

    def _get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get summary of strategy performance"""
        summary = {}

        for strategy_name, performance_history in self.strategy_performance.items():
            if performance_history:
                summary[strategy_name] = {
                    'mean_performance': float(np.mean(performance_history)),
                    'recent_performance': float(np.mean(performance_history[-3:])) if len(performance_history) >= 3 else float(np.mean(performance_history)),
                    'n_uses': len(performance_history),
                    'trend': 'improving' if len(performance_history) >= 2 and performance_history[-1] > performance_history[-2] else 'stable'
                }
            else:
                summary[strategy_name] = {
                    'mean_performance': 0.0,
                    'recent_performance': 0.0,
                    'n_uses': 0,
                    'trend': 'unused'
                }

        return summary

    def _get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            'budget_used': float(self.budget_used),
            'budget_remaining': self._get_remaining_budget(),
            'budget_efficiency': self._calculate_budget_efficiency(),
            'budget_constraints': dict(self.budget_constraints)
        }

    def _get_convergence_indicators(self) -> Dict[str, Any]:
        """Get convergence indicators"""
        return {
            'convergence_trend': self._analyze_convergence_trend(),
            'is_converging': self._detect_convergence(),
            'convergence_history': self.convergence_history[-10:],  # Last 10 values
            'iterations_since_improvement': self._iterations_since_improvement()
        }

    def _iterations_since_improvement(self) -> int:
        """Count iterations since last significant improvement"""
        if not self.convergence_history:
            return 0

        best_value = max(self.convergence_history)
        for i, value in enumerate(reversed(self.convergence_history)):
            if value >= best_value * 0.95:  # Within 5% of best
                return i

        return len(self.convergence_history)

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of adaptive sampling"""
        return {
            'current_strategy': self.current_strategy,
            'total_iterations': self.iteration_count,
            'total_adaptations': len(self.adaptation_decisions),
            'strategy_usage': {
                name: len(performance)
                for name, performance in self.strategy_performance.items()
            },
            'adaptation_frequency_actual': len(self.adaptation_decisions) / max(1, self.iteration_count),
            'best_performing_strategy': max(
                self._rank_strategies().keys(),
                key=lambda k: self._rank_strategies()[k]
            ) if self._rank_strategies() else 'none',
            'convergence_status': {
                'is_converging': self._detect_convergence(),
                'convergence_trend': self._analyze_convergence_trend()
            },
            'budget_efficiency': self._calculate_budget_efficiency()
        }