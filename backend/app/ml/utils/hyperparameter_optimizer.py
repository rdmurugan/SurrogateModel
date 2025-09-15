import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

from ..factory import SurrogateModelFactory


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""

    def __init__(self):
        # Suppress Optuna logs for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    async def optimize(
        self,
        algorithm: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize hyperparameters for a given algorithm.

        Args:
            algorithm: Algorithm name
            X: Input features
            y: Target values
            config: Optimization configuration

        Returns:
            Tuple of (best_hyperparameters, optimization_results)
        """
        # Configuration defaults
        n_trials = config.get('n_trials', 50)
        cv_folds = config.get('cv_folds', 5)
        metric = config.get('metric', 'r2')  # r2, rmse, mae
        timeout = config.get('timeout', 300)  # seconds

        # Create study
        study = optuna.create_study(
            direction='maximize' if metric == 'r2' else 'minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Define objective function
        def objective(trial):
            return self._objective_function(
                trial, algorithm, X, y, metric, cv_folds
            )

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=()
        )

        # Get best hyperparameters
        best_params = study.best_params

        # Prepare results
        optimization_results = {
            'best_value': float(study.best_value),
            'n_trials': len(study.trials),
            'optimization_metric': metric,
            'cv_folds': cv_folds,
            'convergence_history': [trial.value for trial in study.trials if trial.value is not None]
        }

        return best_params, optimization_results

    def _objective_function(
        self,
        trial: optuna.Trial,
        algorithm: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        metric: str,
        cv_folds: int
    ) -> float:
        """Objective function for hyperparameter optimization"""

        try:
            # Get hyperparameters to optimize
            hyperparams = self._suggest_hyperparameters(trial, algorithm)

            # Create model
            model = SurrogateModelFactory.create_model(algorithm, **hyperparams)

            # Prepare data
            X_scaled, y_scaled = model.prepare_data(X, y)

            # Handle multi-output case
            if y_scaled.ndim > 1 and y_scaled.shape[1] > 1:
                # For multi-output, average the scores across outputs
                scores = []
                for i in range(y_scaled.shape[1]):
                    y_target = y_scaled[:, i]
                    target_scores = self._evaluate_model(
                        model, X_scaled, y_target, metric, cv_folds
                    )
                    scores.extend(target_scores)
                return float(np.mean(scores))
            else:
                # Single output
                y_target = y_scaled.ravel() if y_scaled.ndim > 1 else y_scaled
                scores = self._evaluate_model(
                    model, X_scaled, y_target, metric, cv_folds
                )
                return float(np.mean(scores))

        except Exception as e:
            # Return worst possible score for failed trials
            return -1e6 if metric == 'r2' else 1e6

    def _suggest_hyperparameters(self, trial: optuna.Trial, algorithm: str) -> Dict[str, Any]:
        """Suggest hyperparameters for the given algorithm"""

        if algorithm == 'gaussian_process':
            return {
                'kernel_type': trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'rbf_white']),
                'length_scale': trial.suggest_float('length_scale', 0.1, 10.0, log=True),
                'nu': trial.suggest_categorical('nu', [0.5, 1.5, 2.5]),
                'alpha': trial.suggest_float('alpha', 1e-10, 1e-6, log=True),
                'n_restarts_optimizer': trial.suggest_int('n_restarts_optimizer', 5, 20)
            }

        elif algorithm == 'polynomial_chaos':
            return {
                'polynomial_order': trial.suggest_int('polynomial_order', 2, 5),
                'interaction_only': trial.suggest_categorical('interaction_only', [True, False]),
                'sparse_regression': trial.suggest_categorical('sparse_regression', [True, False]),
                'alpha': trial.suggest_float('alpha', 0.001, 1.0, log=True)
            }

        elif algorithm == 'neural_network':
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_layers = []
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 16, 256, log=True)
                hidden_layers.append(size)

            return {
                'hidden_layers': hidden_layers,
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1, log=True),
                'epochs': trial.suggest_int('epochs', 50, 200)
            }

        elif algorithm == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20) if trial.suggest_categorical('use_max_depth', [True, False]) else None,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }

        elif algorithm == 'support_vector':
            return {
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 0.5, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.params.get('kernel') in ['rbf', 'poly'] else 'scale',
                'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3
            }

        elif algorithm == 'radial_basis':
            return {
                'basis_function': trial.suggest_categorical('basis_function', ['gaussian', 'multiquadric', 'inverse_multiquadric']),
                'epsilon': trial.suggest_float('epsilon', 0.1, 5.0),
                'center_selection': trial.suggest_categorical('center_selection', ['data', 'kmeans']),
                'polynomial_degree': trial.suggest_categorical('polynomial_degree', [-1, 0, 1]),
                'regularization': trial.suggest_float('regularization', 1e-12, 1e-6, log=True)
            }

        else:
            return {}

    def _evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
        cv_folds: int
    ) -> np.ndarray:
        """Evaluate model using cross-validation"""

        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create and train model
            temp_model = model._create_model()

            try:
                # Handle different model types
                if hasattr(temp_model, 'fit'):
                    temp_model.fit(X_train, y_train)
                    y_pred = temp_model.predict(X_val)
                else:
                    # For custom models like RBF
                    model._fit_model(X_train, y_train)
                    y_pred = model._predict_model(X_val)

                # Calculate score
                if metric == 'r2':
                    score = r2_score(y_val, y_pred)
                elif metric == 'rmse':
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                elif metric == 'mae':
                    score = np.mean(np.abs(y_val - y_pred))
                else:
                    score = r2_score(y_val, y_pred)

                scores.append(score)

            except Exception:
                # Failed fold gets worst score
                scores.append(-1e6 if metric == 'r2' else 1e6)

        return np.array(scores)


class AutoMLOptimizer:
    """Automated machine learning with algorithm selection and hyperparameter optimization"""

    def __init__(self):
        self.hp_optimizer = HyperparameterOptimizer()

    async def auto_optimize(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Automatically select the best algorithm and optimize its hyperparameters.

        Args:
            X: Input features
            y: Target values
            config: Configuration including algorithms to try

        Returns:
            Dictionary with best algorithm, hyperparameters, and results
        """
        algorithms_to_try = config.get('algorithms', [
            'gaussian_process',
            'random_forest',
            'neural_network',
            'support_vector'
        ])

        time_budget = config.get('time_budget', 600)  # seconds
        trials_per_algorithm = config.get('trials_per_algorithm', 20)

        results = {}
        best_score = -np.inf
        best_algorithm = None
        best_hyperparams = None

        for algorithm in algorithms_to_try:
            try:
                # Optimize this algorithm
                hyperparams, opt_results = await self.hp_optimizer.optimize(
                    algorithm=algorithm,
                    X=X,
                    y=y,
                    config={
                        'n_trials': trials_per_algorithm,
                        'timeout': time_budget // len(algorithms_to_try),
                        **config
                    }
                )

                results[algorithm] = {
                    'best_hyperparameters': hyperparams,
                    'best_score': opt_results['best_value'],
                    'optimization_results': opt_results
                }

                # Track overall best
                if opt_results['best_value'] > best_score:
                    best_score = opt_results['best_value']
                    best_algorithm = algorithm
                    best_hyperparams = hyperparams

            except Exception as e:
                results[algorithm] = {
                    'error': str(e),
                    'best_score': -np.inf
                }

        return {
            'best_algorithm': best_algorithm,
            'best_hyperparameters': best_hyperparams,
            'best_score': float(best_score),
            'all_results': results,
            'recommendation': self._get_recommendation(results)
        }

    def _get_recommendation(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate recommendations based on optimization results"""
        # Sort algorithms by performance
        valid_results = {
            alg: res for alg, res in results.items()
            if 'best_score' in res and res['best_score'] > -np.inf
        }

        if not valid_results:
            return {"message": "No algorithms successfully optimized"}

        sorted_algorithms = sorted(
            valid_results.items(),
            key=lambda x: x[1]['best_score'],
            reverse=True
        )

        top_algorithm = sorted_algorithms[0]
        recommendation = {
            "primary": f"Use {top_algorithm[0]} with score {top_algorithm[1]['best_score']:.4f}",
            "alternatives": []
        }

        # Add alternatives
        for alg, result in sorted_algorithms[1:3]:  # Top 3
            recommendation["alternatives"].append(
                f"{alg}: {result['best_score']:.4f}"
            )

        return recommendation