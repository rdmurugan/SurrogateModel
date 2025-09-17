import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from .acquisition.factory import AcquisitionFunctionFactory
from .sampling_strategies import (
    AdaptiveSampler,
    BatchActiveLearning,
    PhysicsInformedSampler,
    BaseSampler
)
from .multi_fidelity import CoKrigingModel, MultiFidelityModel


class ActiveLearningService:
    """
    Comprehensive service for orchestrating active learning workflows.

    This service provides a high-level interface for:
    1. Coordinating different sampling strategies
    2. Managing model updates and retraining
    3. Handling multi-fidelity scenarios
    4. Budget and resource management
    5. Performance monitoring and optimization
    6. Asynchronous execution for scalability
    """

    def __init__(self,
                 model_config: Dict[str, Any],
                 sampling_config: Dict[str, Any] = None,
                 budget_config: Dict[str, Any] = None,
                 performance_config: Dict[str, Any] = None):
        """
        Initialize the Active Learning Service.

        Args:
            model_config: Configuration for the surrogate model
            sampling_config: Configuration for sampling strategies
            budget_config: Budget and resource constraints
            performance_config: Performance monitoring configuration
        """
        self.logger = logging.getLogger(__name__)

        # Model management
        self.model_config = model_config
        self.primary_model = None
        self.multi_fidelity_model = None

        # Sampling strategy management
        self.sampling_config = sampling_config or {}
        self.adaptive_sampler = None
        self.specialized_samplers = {}

        # Budget and resource management
        self.budget_config = budget_config or {}
        self.budget_tracker = BudgetTracker(budget_config)

        # Performance monitoring
        self.performance_config = performance_config or {}
        self.performance_monitor = PerformanceMonitor(performance_config)

        # Active learning state
        self.iteration_count = 0
        self.training_history = []
        self.sampling_history = []
        self.is_initialized = False

        # Asynchronous execution
        self.executor = ThreadPoolExecutor(max_workers=4)

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all service components"""
        try:
            # Initialize primary model
            self._initialize_model()

            # Initialize sampling strategies
            self._initialize_samplers()

            # Initialize multi-fidelity if configured
            self._initialize_multi_fidelity()

            self.is_initialized = True
            self.logger.info("Active Learning Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Active Learning Service: {e}")
            raise

    def _initialize_model(self):
        """Initialize the primary surrogate model"""
        model_type = self.model_config.get('type', 'gaussian_process')
        model_params = self.model_config.get('params', {})

        self.primary_model = ModelFactory.create(model_type, **model_params)

    def _initialize_samplers(self):
        """Initialize sampling strategies"""
        # Adaptive sampler (main orchestrator)
        adaptive_config = self.sampling_config.get('adaptive', {})
        self.adaptive_sampler = AdaptiveSampler(
            strategies=adaptive_config.get('strategies'),
            adaptation_frequency=adaptive_config.get('adaptation_frequency', 5),
            performance_window=adaptive_config.get('performance_window', 10),
            budget_constraints=self.budget_config,
            convergence_threshold=adaptive_config.get('convergence_threshold', 1e-4)
        )

        # Specialized samplers
        if 'batch' in self.sampling_config:
            batch_config = self.sampling_config['batch']
            self.specialized_samplers['batch'] = BatchActiveLearning(**batch_config)

        if 'physics_informed' in self.sampling_config:
            physics_config = self.sampling_config['physics_informed']
            self.specialized_samplers['physics_informed'] = PhysicsInformedSampler(**physics_config)

    def _initialize_multi_fidelity(self):
        """Initialize multi-fidelity modeling if configured"""
        if 'multi_fidelity' in self.model_config:
            mf_config = self.model_config['multi_fidelity']
            fidelity_levels = mf_config.get('fidelity_levels', [])

            if fidelity_levels:
                if mf_config.get('type', 'co_kriging') == 'co_kriging':
                    self.multi_fidelity_model = CoKrigingModel(
                        fidelity_levels=fidelity_levels,
                        correlation_prior=mf_config.get('correlation_prior', 0.8)
                    )

    async def start_active_learning(self,
                                  initial_data: Dict[str, np.ndarray],
                                  candidate_points: np.ndarray,
                                  max_iterations: int = 50,
                                  convergence_criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Start the active learning process.

        Args:
            initial_data: Initial training data {'X': features, 'y': targets}
            candidate_points: Pool of candidate points for sampling
            max_iterations: Maximum number of AL iterations
            convergence_criteria: Criteria for early stopping

        Returns:
            Complete active learning results
        """
        if not self.is_initialized:
            raise RuntimeError("Service not properly initialized")

        self.logger.info(f"Starting active learning with {len(initial_data['X'])} initial samples")

        # Initialize with initial data
        await self._initialize_with_data(initial_data)

        # Main active learning loop
        results = await self._active_learning_loop(
            candidate_points, max_iterations, convergence_criteria
        )

        return results

    async def _initialize_with_data(self, initial_data: Dict[str, np.ndarray]):
        """Initialize models with initial training data"""
        X_init = initial_data['X']
        y_init = initial_data['y']

        # Train primary model
        await self._train_model_async(self.primary_model, X_init, y_init)

        # Initialize multi-fidelity if available
        if self.multi_fidelity_model is not None:
            # Assume initial data is highest fidelity
            highest_fidelity = self.multi_fidelity_model.get_highest_fidelity()
            mf_data = {highest_fidelity.level: (X_init, y_init)}
            await self._train_multi_fidelity_async(mf_data)

        # Record initial state
        self.training_history.append({
            'iteration': 0,
            'n_samples': len(X_init),
            'model_performance': await self._evaluate_model_performance_async(X_init, y_init),
            'timestamp': datetime.now()
        })

    async def _active_learning_loop(self,
                                  candidate_points: np.ndarray,
                                  max_iterations: int,
                                  convergence_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Main active learning loop"""
        converged = False
        remaining_candidates = candidate_points.copy()

        for iteration in range(1, max_iterations + 1):
            self.iteration_count = iteration

            self.logger.info(f"Active Learning Iteration {iteration}/{max_iterations}")

            # Check convergence
            if await self._check_convergence(convergence_criteria):
                self.logger.info(f"Convergence achieved at iteration {iteration}")
                converged = True
                break

            # Check budget constraints
            if not self.budget_tracker.can_continue():
                self.logger.info("Budget exhausted, stopping active learning")
                break

            # Sample new points
            sampling_result = await self._execute_sampling_iteration(remaining_candidates)

            if sampling_result is None or len(sampling_result['selected_points']) == 0:
                self.logger.warning("No points selected, stopping active learning")
                break

            # Simulate or execute experiments
            new_data = await self._execute_experiments(sampling_result['selected_points'])

            # Update models
            await self._update_models(new_data)

            # Update candidates (remove selected points)
            remaining_candidates = self._update_candidate_pool(
                remaining_candidates, sampling_result['selected_indices']
            )

            # Record iteration results
            await self._record_iteration_results(iteration, sampling_result, new_data)

            # Update budget
            self.budget_tracker.update_usage(sampling_result, new_data)

            if len(remaining_candidates) == 0:
                self.logger.info("Candidate pool exhausted")
                break

        # Compile final results
        final_results = await self._compile_final_results(converged)

        return final_results

    async def _execute_sampling_iteration(self, candidate_points: np.ndarray) -> Optional[Dict[str, Any]]:
        """Execute one sampling iteration"""
        try:
            # Determine number of samples based on budget and strategy
            n_samples = self._determine_sample_size(candidate_points)

            if n_samples == 0:
                return None

            # Use adaptive sampler for point selection
            sampling_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.adaptive_sampler.sample,
                self.primary_model,
                candidate_points,
                n_samples
            )

            # Record sampling decision
            self.sampling_history.append({
                'iteration': self.iteration_count,
                'strategy_used': sampling_result.get('adaptive_info', {}).get('current_strategy', 'unknown'),
                'n_samples': n_samples,
                'timestamp': datetime.now()
            })

            return sampling_result

        except Exception as e:
            self.logger.error(f"Sampling iteration failed: {e}")
            return None

    async def _execute_experiments(self, selected_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute experiments (simulation or actual) for selected points"""
        # In a real implementation, this would interface with simulation engines
        # For now, we'll simulate the process

        self.logger.info(f"Executing experiments for {len(selected_points)} points")

        # Simulate experiment execution with some delay
        await asyncio.sleep(0.1)

        # For demonstration, use the model to generate synthetic data with noise
        if hasattr(self.primary_model, 'predict'):
            try:
                predictions = self.primary_model.predict(selected_points)

                if isinstance(predictions, dict):
                    # Extract predictions from structured output
                    y_values = []
                    for pred in predictions.values():
                        if isinstance(pred, dict) and 'prediction' in pred:
                            y_values.append(pred['prediction'])
                    y_new = np.array(y_values)
                else:
                    y_new = predictions

                # Add realistic noise
                noise_level = 0.1
                noise = np.random.normal(0, noise_level, size=y_new.shape)
                y_new += noise

            except Exception as e:
                self.logger.warning(f"Failed to get model predictions for synthetic data: {e}")
                # Fallback to random data
                y_new = np.random.normal(0, 1, size=(len(selected_points),))

        else:
            # Fallback to random data
            y_new = np.random.normal(0, 1, size=(len(selected_points),))

        return {
            'X': selected_points,
            'y': y_new
        }

    async def _update_models(self, new_data: Dict[str, np.ndarray]):
        """Update all models with new data"""
        X_new = new_data['X']
        y_new = new_data['y']

        # Get existing training data
        if hasattr(self.primary_model, 'training_data'):
            X_existing, y_existing = self.primary_model.training_data
        else:
            # Initialize if no training data exists
            X_existing, y_existing = np.array([]).reshape(0, X_new.shape[1]), np.array([])

        # Combine existing and new data
        X_combined = np.vstack([X_existing, X_new]) if len(X_existing) > 0 else X_new
        y_combined = np.hstack([y_existing, y_new]) if len(y_existing) > 0 else y_new

        # Retrain primary model
        await self._train_model_async(self.primary_model, X_combined, y_combined)

        # Update multi-fidelity model if available
        if self.multi_fidelity_model is not None:
            await self._update_multi_fidelity_model(X_new, y_new)

    async def _train_model_async(self, model, X: np.ndarray, y: np.ndarray):
        """Train model asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, model.fit, X, y)

        # Store training data reference
        model.training_data = (X, y)

    async def _train_multi_fidelity_async(self, multi_fidelity_data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """Train multi-fidelity model asynchronously"""
        if self.multi_fidelity_model is None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.multi_fidelity_model.fit, multi_fidelity_data)

    async def _update_multi_fidelity_model(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update multi-fidelity model with new high-fidelity data"""
        if self.multi_fidelity_model is None:
            return

        highest_fidelity = self.multi_fidelity_model.get_highest_fidelity()
        X_existing, y_existing = highest_fidelity.get_all_data()

        X_combined = np.vstack([X_existing, X_new]) if len(X_existing) > 0 else X_new
        y_combined = np.hstack([y_existing, y_new]) if len(y_existing) > 0 else y_new

        # Get low fidelity data
        lowest_fidelity = self.multi_fidelity_model.get_lowest_fidelity()
        X_low, y_low = lowest_fidelity.get_all_data()

        mf_data = {
            lowest_fidelity.level: (X_low, y_low),
            highest_fidelity.level: (X_combined, y_combined)
        }

        await self._train_multi_fidelity_async(mf_data)

    def _determine_sample_size(self, candidate_points: np.ndarray) -> int:
        """Determine appropriate sample size based on budget and strategy"""
        # Budget constraints
        max_samples_budget = self.budget_tracker.get_max_samples_for_iteration()

        # Strategy preferences
        max_samples_strategy = self.sampling_config.get('max_samples_per_iteration', 5)

        # Candidate availability
        max_samples_available = len(candidate_points)

        # Take minimum of all constraints
        n_samples = min(max_samples_budget, max_samples_strategy, max_samples_available)

        return max(0, n_samples)

    def _update_candidate_pool(self, candidates: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
        """Remove selected points from candidate pool"""
        remaining_mask = np.ones(len(candidates), dtype=bool)
        remaining_mask[selected_indices] = False
        return candidates[remaining_mask]

    async def _check_convergence(self, convergence_criteria: Dict[str, Any]) -> bool:
        """Check if convergence criteria are met"""
        if not convergence_criteria:
            return False

        # Model performance based convergence
        if 'model_improvement_threshold' in convergence_criteria:
            if len(self.training_history) >= 2:
                recent_improvement = self._calculate_recent_improvement()
                threshold = convergence_criteria['model_improvement_threshold']
                if recent_improvement < threshold:
                    return True

        # Budget based convergence
        if 'budget_threshold' in convergence_criteria:
            remaining_budget = self.budget_tracker.get_remaining_budget_ratio()
            threshold = convergence_criteria['budget_threshold']
            if remaining_budget < threshold:
                return True

        # Iteration based convergence
        if 'max_iterations_without_improvement' in convergence_criteria:
            max_iterations = convergence_criteria['max_iterations_without_improvement']
            iterations_without_improvement = self._count_iterations_without_improvement()
            if iterations_without_improvement >= max_iterations:
                return True

        return False

    def _calculate_recent_improvement(self) -> float:
        """Calculate recent model performance improvement"""
        if len(self.training_history) < 2:
            return float('inf')

        recent_performance = self.training_history[-1]['model_performance'].get('r2_score', 0)
        previous_performance = self.training_history[-2]['model_performance'].get('r2_score', 0)

        return recent_performance - previous_performance

    def _count_iterations_without_improvement(self) -> int:
        """Count iterations without significant improvement"""
        if len(self.training_history) < 2:
            return 0

        best_performance = max(
            entry['model_performance'].get('r2_score', 0)
            for entry in self.training_history
        )

        iterations_without_improvement = 0
        for entry in reversed(self.training_history):
            current_performance = entry['model_performance'].get('r2_score', 0)
            if current_performance < best_performance * 0.95:  # Within 5% of best
                iterations_without_improvement += 1
            else:
                break

        return iterations_without_improvement

    async def _evaluate_model_performance_async(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._evaluate_model_performance, X, y)

    def _evaluate_model_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import r2_score, mean_squared_error

            # Cross-validation score
            if hasattr(self.primary_model, 'predict') and len(X) > 3:
                cv_scores = cross_val_score(self.primary_model, X, y, cv=min(5, len(X)), scoring='r2')
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
            else:
                cv_mean, cv_std = 0.0, 0.0

            # Prediction metrics
            if hasattr(self.primary_model, 'predict'):
                y_pred = self.primary_model.predict(X)

                if isinstance(y_pred, dict):
                    # Extract predictions from structured output
                    y_pred_values = []
                    for pred in y_pred.values():
                        if isinstance(pred, dict) and 'prediction' in pred:
                            y_pred_values.append(pred['prediction'])
                    y_pred = np.array(y_pred_values) if y_pred_values else np.zeros_like(y)

                r2 = float(r2_score(y, y_pred))
                mse = float(mean_squared_error(y, y_pred))
            else:
                r2, mse = 0.0, float('inf')

            return {
                'r2_score': r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'mse': mse,
                'n_samples': len(X)
            }

        except Exception as e:
            self.logger.warning(f"Model performance evaluation failed: {e}")
            return {
                'r2_score': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'mse': float('inf'),
                'n_samples': len(X)
            }

    async def _record_iteration_results(self,
                                      iteration: int,
                                      sampling_result: Dict[str, Any],
                                      new_data: Dict[str, np.ndarray]):
        """Record results from the current iteration"""
        # Get current training data
        if hasattr(self.primary_model, 'training_data'):
            X_current, y_current = self.primary_model.training_data
        else:
            X_current, y_current = new_data['X'], new_data['y']

        # Evaluate model performance
        model_performance = await self._evaluate_model_performance_async(X_current, y_current)

        # Record training history
        self.training_history.append({
            'iteration': iteration,
            'n_samples': len(X_current),
            'new_samples': len(new_data['X']),
            'model_performance': model_performance,
            'sampling_strategy': sampling_result.get('adaptive_info', {}).get('current_strategy', 'unknown'),
            'timestamp': datetime.now()
        })

        # Update performance monitor
        self.performance_monitor.update(iteration, sampling_result, model_performance)

    async def _compile_final_results(self, converged: bool) -> Dict[str, Any]:
        """Compile final active learning results"""
        # Get final training data
        if hasattr(self.primary_model, 'training_data'):
            X_final, y_final = self.primary_model.training_data
        else:
            X_final, y_final = np.array([]), np.array([])

        final_performance = await self._evaluate_model_performance_async(X_final, y_final)

        results = {
            'success': True,
            'converged': converged,
            'total_iterations': self.iteration_count,
            'final_sample_count': len(X_final),
            'final_performance': final_performance,
            'training_history': self.training_history,
            'sampling_history': self.sampling_history,
            'budget_summary': self.budget_tracker.get_summary(),
            'performance_summary': self.performance_monitor.get_summary(),
            'adaptive_sampling_summary': self.adaptive_sampler.get_adaptation_summary(),
            'execution_time': {
                'start_time': self.training_history[0]['timestamp'] if self.training_history else datetime.now(),
                'end_time': datetime.now()
            }
        }

        # Multi-fidelity analysis if available
        if self.multi_fidelity_model is not None:
            results['multi_fidelity_analysis'] = self.multi_fidelity_model.get_correlation_analysis()

        return results

    async def get_model_predictions(self, X: np.ndarray) -> Dict[str, Any]:
        """Get predictions from the trained model"""
        if not self.is_initialized or self.primary_model is None:
            raise RuntimeError("Model not properly initialized")

        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(self.executor, self.primary_model.predict, X)

        return {
            'predictions': predictions,
            'model_type': self.model_config.get('type', 'unknown'),
            'training_samples': len(self.primary_model.training_data[0]) if hasattr(self.primary_model, 'training_data') else 0
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'is_initialized': self.is_initialized,
            'current_iteration': self.iteration_count,
            'total_training_samples': len(self.primary_model.training_data[0]) if hasattr(self.primary_model, 'training_data') else 0,
            'budget_status': self.budget_tracker.get_summary() if self.budget_tracker else {},
            'performance_status': self.performance_monitor.get_summary() if self.performance_monitor else {},
            'adaptive_sampler_status': self.adaptive_sampler.get_adaptation_summary() if self.adaptive_sampler else {}
        }

    async def cleanup(self):
        """Cleanup service resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

        self.logger.info("Active Learning Service cleaned up successfully")


class BudgetTracker:
    """Track and manage budget constraints for active learning"""

    def __init__(self, budget_config: Dict[str, Any]):
        self.total_budget = budget_config.get('total_budget', 1000.0)
        self.cost_per_sample = budget_config.get('cost_per_sample', 1.0)
        self.used_budget = 0.0
        self.usage_history = []

    def can_continue(self) -> bool:
        """Check if budget allows continuation"""
        return self.used_budget < self.total_budget

    def get_max_samples_for_iteration(self) -> int:
        """Get maximum samples allowable for current iteration"""
        remaining_budget = self.total_budget - self.used_budget
        max_samples = int(remaining_budget / self.cost_per_sample)
        return max(0, max_samples)

    def update_usage(self, sampling_result: Dict[str, Any], new_data: Dict[str, np.ndarray]):
        """Update budget usage"""
        n_samples = len(new_data['X'])
        cost = n_samples * self.cost_per_sample
        self.used_budget += cost

        self.usage_history.append({
            'iteration': len(self.usage_history) + 1,
            'samples': n_samples,
            'cost': cost,
            'cumulative_cost': self.used_budget,
            'timestamp': datetime.now()
        })

    def get_remaining_budget_ratio(self) -> float:
        """Get remaining budget as ratio of total"""
        return (self.total_budget - self.used_budget) / self.total_budget

    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary"""
        return {
            'total_budget': self.total_budget,
            'used_budget': self.used_budget,
            'remaining_budget': self.total_budget - self.used_budget,
            'budget_efficiency': self.used_budget / self.total_budget if self.total_budget > 0 else 0,
            'usage_history': self.usage_history
        }


class PerformanceMonitor:
    """Monitor and analyze active learning performance"""

    def __init__(self, performance_config: Dict[str, Any]):
        self.metrics = []
        self.strategy_performance = {}
        self.improvement_trend = []

    def update(self, iteration: int, sampling_result: Dict[str, Any], model_performance: Dict[str, float]):
        """Update performance metrics"""
        strategy = sampling_result.get('adaptive_info', {}).get('current_strategy', 'unknown')

        metric_entry = {
            'iteration': iteration,
            'strategy': strategy,
            'model_performance': model_performance,
            'timestamp': datetime.now()
        }

        self.metrics.append(metric_entry)

        # Track strategy-specific performance
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []

        self.strategy_performance[strategy].append(model_performance.get('r2_score', 0))

        # Calculate improvement trend
        if len(self.metrics) >= 2:
            current_r2 = model_performance.get('r2_score', 0)
            previous_r2 = self.metrics[-2]['model_performance'].get('r2_score', 0)
            improvement = current_r2 - previous_r2
            self.improvement_trend.append(improvement)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {}

        # Overall performance
        r2_scores = [m['model_performance'].get('r2_score', 0) for m in self.metrics]
        final_r2 = r2_scores[-1] if r2_scores else 0
        improvement = final_r2 - r2_scores[0] if len(r2_scores) > 1 else 0

        # Strategy analysis
        strategy_summary = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_summary[strategy] = {
                    'mean_performance': float(np.mean(performances)),
                    'best_performance': float(np.max(performances)),
                    'n_uses': len(performances)
                }

        return {
            'final_r2_score': final_r2,
            'total_improvement': improvement,
            'mean_improvement_per_iteration': float(np.mean(self.improvement_trend)) if self.improvement_trend else 0,
            'strategy_performance': strategy_summary,
            'convergence_rate': self._calculate_convergence_rate()
        }

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on improvement trend"""
        if len(self.improvement_trend) < 3:
            return 0.0

        # Fit exponential decay to improvement trend
        recent_improvements = self.improvement_trend[-10:]  # Last 10 improvements
        if all(imp <= 0.001 for imp in recent_improvements):
            return 1.0  # Converged

        return 1.0 - (len([imp for imp in recent_improvements if imp > 0.001]) / len(recent_improvements))