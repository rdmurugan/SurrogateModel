import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats


class ModelValidator:
    """Comprehensive model validation and diagnostics"""

    def __init__(self):
        self.validation_results = {}

    def validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.DataFrame,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model validation.

        Args:
            model: Trained surrogate model
            X: Input features
            y: Target values
            validation_config: Configuration for validation

        Returns:
            Dictionary with validation results and metrics
        """
        config = validation_config or {}
        test_size = config.get('test_size', 0.2)
        cv_folds = config.get('cv_folds', 5)
        include_plots = config.get('include_plots', False)

        results = {}

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 1. Cross-validation metrics
        cv_results = self._cross_validation_metrics(model, X, y, cv_folds)
        results['cross_validation'] = cv_results

        # 2. Hold-out test metrics
        test_results = self._holdout_test_metrics(model, X_test, y_test)
        results['holdout_test'] = test_results

        # 3. Residual analysis
        residual_results = self._residual_analysis(model, X_test, y_test)
        results['residual_analysis'] = residual_results

        # 4. Model diagnostics
        diagnostic_results = self._model_diagnostics(model, X_test, y_test)
        results['diagnostics'] = diagnostic_results

        # 5. Uncertainty calibration (if model supports uncertainty)
        if hasattr(model, 'get_uncertainty'):
            uncertainty_results = self._uncertainty_calibration(model, X_test, y_test)
            results['uncertainty_calibration'] = uncertainty_results

        # 6. Feature importance and sensitivity
        sensitivity_results = self._sensitivity_analysis(model, X, y)
        results['sensitivity_analysis'] = sensitivity_results

        # 7. Generate plots if requested
        if include_plots:
            plots = self._generate_validation_plots(model, X_test, y_test)
            results['plots'] = plots

        return results

    def _cross_validation_metrics(self, model, X: pd.DataFrame, y: pd.DataFrame, cv_folds: int) -> Dict[str, Any]:
        """Perform cross-validation and calculate metrics"""
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_results = {}
        for target_idx, target_name in enumerate(model.target_names):
            y_target = y.iloc[:, target_idx] if len(model.target_names) > 1 else y.iloc[:, 0]

            # Calculate R² scores
            r2_scores = []
            rmse_scores = []
            mae_scores = []

            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y_target.iloc[train_idx], y_target.iloc[val_idx]

                # Create temporary model for this fold
                temp_model = model.__class__(**model.hyperparameters)
                temp_model.fit(X_train_fold, pd.DataFrame({target_name: y_train_fold}))

                # Make predictions
                pred_result = temp_model.predict(X_val_fold)
                y_pred = [pred_result[target_name]['prediction'] for _ in range(len(X_val_fold))]

                # Calculate metrics
                r2_scores.append(r2_score(y_val_fold, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
                mae_scores.append(mean_absolute_error(y_val_fold, y_pred))

            cv_results[target_name] = {
                'r2_mean': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'rmse_mean': float(np.mean(rmse_scores)),
                'rmse_std': float(np.std(rmse_scores)),
                'mae_mean': float(np.mean(mae_scores)),
                'mae_std': float(np.std(mae_scores))
            }

        return cv_results

    def _holdout_test_metrics(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics on hold-out test set"""
        test_results = {}

        for target_idx, target_name in enumerate(model.target_names):
            y_true = y_test.iloc[:, target_idx] if len(model.target_names) > 1 else y_test.iloc[:, 0]

            # Make predictions
            predictions = []
            for i in range(len(X_test)):
                pred_result = model.predict(X_test.iloc[i:i+1])
                predictions.append(pred_result[target_name]['prediction'])

            y_pred = np.array(predictions)

            # Calculate comprehensive metrics
            test_results[target_name] = {
                'r2_score': float(r2_score(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
                'max_error': float(np.max(np.abs(y_true - y_pred))),
                'median_error': float(np.median(np.abs(y_true - y_pred))),
                'explained_variance': float(1 - np.var(y_true - y_pred) / np.var(y_true))
            }

        return test_results

    def _residual_analysis(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """Analyze residuals for model diagnostics"""
        residual_results = {}

        for target_idx, target_name in enumerate(model.target_names):
            y_true = y_test.iloc[:, target_idx] if len(model.target_names) > 1 else y_test.iloc[:, 0]

            # Make predictions
            predictions = []
            for i in range(len(X_test)):
                pred_result = model.predict(X_test.iloc[i:i+1])
                predictions.append(pred_result[target_name]['prediction'])

            y_pred = np.array(predictions)
            residuals = y_true - y_pred

            # Statistical tests on residuals
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            _, durbin_watson = self._durbin_watson_statistic(residuals)

            residual_results[target_name] = {
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_skewness': float(stats.skew(residuals)),
                'residual_kurtosis': float(stats.kurtosis(residuals)),
                'shapiro_wilk_stat': float(shapiro_stat),
                'shapiro_wilk_p_value': float(shapiro_p),
                'normality_test_passed': shapiro_p > 0.05,
                'durbin_watson_statistic': float(durbin_watson),
                'autocorrelation_detected': durbin_watson < 1.5 or durbin_watson > 2.5
            }

        return residual_results

    def _model_diagnostics(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """Perform model-specific diagnostics"""
        diagnostics = {
            'model_type': model.__class__.__name__,
            'n_features': len(model.feature_names),
            'n_targets': len(model.target_names),
            'is_trained': model.is_trained
        }

        # Add model-specific diagnostics
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance:
                diagnostics['feature_importance'] = feature_importance
                diagnostics['most_important_feature'] = max(feature_importance.items(), key=lambda x: x[1])[0]

        # Check for potential overfitting
        train_r2 = np.mean([metrics['r2'] for metrics in model.training_metrics.values() if 'r2' in metrics])
        test_metrics = self._holdout_test_metrics(model, X_test, y_test)
        test_r2 = np.mean([metrics['r2_score'] for metrics in test_metrics.values()])

        diagnostics['potential_overfitting'] = (train_r2 - test_r2) > 0.1
        diagnostics['generalization_gap'] = float(train_r2 - test_r2)

        return diagnostics

    def _uncertainty_calibration(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """Assess uncertainty calibration"""
        calibration_results = {}

        for target_idx, target_name in enumerate(model.target_names):
            y_true = y_test.iloc[:, target_idx] if len(model.target_names) > 1 else y_test.iloc[:, 0]

            # Get predictions with uncertainty
            predictions = []
            uncertainties = []

            for i in range(len(X_test)):
                pred_result = model.predict(X_test.iloc[i:i+1])
                predictions.append(pred_result[target_name]['prediction'])
                uncertainties.append(pred_result[target_name]['uncertainty']['standard_deviation'])

            y_pred = np.array(predictions)
            std_pred = np.array(uncertainties)

            # Calculate normalized residuals
            normalized_residuals = (y_true - y_pred) / (std_pred + 1e-8)

            # Check if residuals follow standard normal distribution
            shapiro_stat, shapiro_p = stats.shapiro(normalized_residuals)

            # Calculate coverage probabilities
            coverage_68 = np.mean(np.abs(normalized_residuals) <= 1.0)  # Should be ~0.68
            coverage_95 = np.mean(np.abs(normalized_residuals) <= 1.96)  # Should be ~0.95

            calibration_results[target_name] = {
                'normalized_residual_mean': float(np.mean(normalized_residuals)),
                'normalized_residual_std': float(np.std(normalized_residuals)),
                'coverage_68_percent': float(coverage_68),
                'coverage_95_percent': float(coverage_95),
                'well_calibrated_68': abs(coverage_68 - 0.68) < 0.05,
                'well_calibrated_95': abs(coverage_95 - 0.95) < 0.05,
                'uncertainty_quality_score': float(1.0 - abs(coverage_95 - 0.95))
            }

        return calibration_results

    def _sensitivity_analysis(self, model, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Perform global sensitivity analysis"""
        sensitivity_results = {}

        # Simple sensitivity analysis using feature perturbation
        baseline_pred = model.predict(X.iloc[:1])

        for feature_name in model.feature_names:
            feature_sensitivities = {}

            for target_name in model.target_names:
                baseline_value = baseline_pred[target_name]['prediction']

                # Perturb feature by ±10%
                X_perturbed = X.iloc[:1].copy()
                original_value = X_perturbed[feature_name].iloc[0]

                # Positive perturbation
                X_perturbed[feature_name] = original_value * 1.1
                pred_pos = model.predict(X_perturbed)
                change_pos = pred_pos[target_name]['prediction'] - baseline_value

                # Negative perturbation
                X_perturbed[feature_name] = original_value * 0.9
                pred_neg = model.predict(X_perturbed)
                change_neg = baseline_value - pred_neg[target_name]['prediction']

                # Calculate sensitivity
                sensitivity = (change_pos + change_neg) / (2 * 0.1 * abs(original_value))
                feature_sensitivities[target_name] = float(sensitivity)

            sensitivity_results[feature_name] = feature_sensitivities

        return sensitivity_results

    def _generate_validation_plots(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, str]:
        """Generate validation plots as base64 encoded strings"""
        plots = {}

        for target_idx, target_name in enumerate(model.target_names):
            y_true = y_test.iloc[:, target_idx] if len(model.target_names) > 1 else y_test.iloc[:, 0]

            # Make predictions
            predictions = []
            for i in range(len(X_test)):
                pred_result = model.predict(X_test.iloc[i:i+1])
                predictions.append(pred_result[target_name]['prediction'])

            y_pred = np.array(predictions)

            # Create prediction vs actual plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Prediction vs Actual - {target_name}')
            plt.grid(True, alpha=0.3)

            # Save plot as base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots[f'{target_name}_prediction_plot'] = plot_data
            plt.close()

            # Create residual plot
            residuals = y_true - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residual Plot - {target_name}')
            plt.grid(True, alpha=0.3)

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots[f'{target_name}_residual_plot'] = plot_data
            plt.close()

        return plots

    def _durbin_watson_statistic(self, residuals: np.ndarray) -> Tuple[str, float]:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        diff_residuals = np.diff(residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)

        if dw_stat < 1.5:
            interpretation = "positive_autocorrelation"
        elif dw_stat > 2.5:
            interpretation = "negative_autocorrelation"
        else:
            interpretation = "no_autocorrelation"

        return interpretation, dw_stat

    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("# Model Validation Report\n")

        # Overall summary
        report.append("## Summary")
        cv_results = validation_results.get('cross_validation', {})
        test_results = validation_results.get('holdout_test', {})

        for target_name in cv_results.keys():
            cv_r2 = cv_results[target_name]['r2_mean']
            test_r2 = test_results[target_name]['r2_score']
            report.append(f"- **{target_name}**: CV R² = {cv_r2:.4f}, Test R² = {test_r2:.4f}")

        # Diagnostics
        report.append("\n## Diagnostics")
        diagnostics = validation_results.get('diagnostics', {})

        if diagnostics.get('potential_overfitting'):
            report.append("⚠️ **Warning**: Potential overfitting detected")

        if 'feature_importance' in diagnostics:
            report.append(f"- Most important feature: {diagnostics['most_important_feature']}")

        # Residual analysis
        report.append("\n## Residual Analysis")
        residual_analysis = validation_results.get('residual_analysis', {})

        for target_name, residuals in residual_analysis.items():
            if not residuals['normality_test_passed']:
                report.append(f"⚠️ **{target_name}**: Residuals may not be normally distributed")

            if residuals['autocorrelation_detected']:
                report.append(f"⚠️ **{target_name}**: Autocorrelation detected in residuals")

        # Uncertainty calibration
        if 'uncertainty_calibration' in validation_results:
            report.append("\n## Uncertainty Calibration")
            uncertainty_cal = validation_results['uncertainty_calibration']

            for target_name, cal_results in uncertainty_cal.items():
                quality_score = cal_results['uncertainty_quality_score']
                if quality_score > 0.9:
                    status = "✅ Excellent"
                elif quality_score > 0.8:
                    status = "🟡 Good"
                else:
                    status = "❌ Poor"

                report.append(f"- **{target_name}**: {status} (Quality Score: {quality_score:.3f})")

        return "\n".join(report)