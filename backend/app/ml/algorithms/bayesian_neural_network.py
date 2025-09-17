import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BayesianLayer(nn.Module):
    """
    Bayesian linear layer with variational inference.

    Uses reparameterization trick for gradient estimation.
    Maintains weight and bias distributions instead of point estimates.
    """

    def __init__(self, input_size: int, output_size: int, prior_std: float = 1.0):
        super(BayesianLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.prior_std = prior_std

        # Weight parameters (mean and log std)
        self.weight_mu = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.weight_log_std = nn.Parameter(torch.randn(output_size, input_size) * 0.1 - 2)

        # Bias parameters (mean and log std)
        self.bias_mu = nn.Parameter(torch.randn(output_size) * 0.1)
        self.bias_log_std = nn.Parameter(torch.randn(output_size) * 0.1 - 2)

        # Prior distributions
        self.weight_prior = Normal(0, prior_std)
        self.bias_prior = Normal(0, prior_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        # Sample weights
        weight_std = torch.exp(self.weight_log_std)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std

        # Sample biases
        bias_std = torch.exp(self.bias_log_std)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Weight KL divergence
        weight_posterior = Normal(self.weight_mu, torch.exp(self.weight_log_std))
        weight_kl = kl_divergence(weight_posterior, self.weight_prior).sum()

        # Bias KL divergence
        bias_posterior = Normal(self.bias_mu, torch.exp(self.bias_log_std))
        bias_kl = kl_divergence(bias_posterior, self.bias_prior).sum()

        return weight_kl + bias_kl


class MCDropoutLayer(nn.Module):
    """
    Monte Carlo Dropout layer that remains active during inference.

    Provides uncertainty estimation through stochastic forward passes.
    """

    def __init__(self, dropout_rate: float = 0.1, concrete_dropout: bool = False):
        super(MCDropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.concrete_dropout = concrete_dropout

        if concrete_dropout:
            # Learnable dropout rate
            self.p_logit = nn.Parameter(torch.tensor(np.log(dropout_rate / (1 - dropout_rate))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode"""
        if self.concrete_dropout:
            # Concrete dropout with learnable rate
            p = torch.sigmoid(self.p_logit)
            return F.dropout(x, p, training=True)
        else:
            # Standard MC dropout
            return F.dropout(x, self.dropout_rate, training=True)

    def get_dropout_rate(self) -> float:
        """Get current dropout rate"""
        if self.concrete_dropout:
            return torch.sigmoid(self.p_logit).item()
        return self.dropout_rate


class BayesianNeuralNetwork(nn.Module):
    """
    Bayesian Neural Network with variational inference.

    Combines Bayesian layers with uncertainty quantification capabilities.
    Supports both aleatoric and epistemic uncertainty estimation.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: List[int] = [100, 100],
                 activation: str = 'relu',
                 use_mc_dropout: bool = True,
                 dropout_rate: float = 0.1,
                 prior_std: float = 1.0,
                 heteroscedastic: bool = True):
        """
        Initialize Bayesian Neural Network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'swish')
            use_mc_dropout: Whether to use MC dropout
            dropout_rate: Dropout rate for MC dropout
            prior_std: Standard deviation for weight priors
            heteroscedastic: Whether to model aleatoric uncertainty
        """
        super(BayesianNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_mc_dropout = use_mc_dropout
        self.heteroscedastic = heteroscedastic

        # Create layers
        layers = []
        layer_sizes = [input_dim] + hidden_layers

        for i in range(len(layer_sizes) - 1):
            # Bayesian linear layer
            layers.append(BayesianLayer(layer_sizes[i], layer_sizes[i + 1], prior_std))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'swish':
                layers.append(nn.SiLU())

            # MC Dropout
            if use_mc_dropout:
                layers.append(MCDropoutLayer(dropout_rate))

        # Output layer
        if heteroscedastic:
            # Output both mean and log variance
            layers.append(BayesianLayer(layer_sizes[-1], output_dim * 2, prior_std))
        else:
            # Output only mean
            layers.append(BayesianLayer(layer_sizes[-1], output_dim, prior_std))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Bayesian network"""
        for layer in self.layers:
            x = layer(x)
        return x

    def predict_with_uncertainty(self, x: torch.Tensor,
                                n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification.

        Returns:
            mean: Predictive mean
            aleatoric_std: Aleatoric (data) uncertainty
            epistemic_std: Epistemic (model) uncertainty
        """
        self.train()  # Enable dropout

        predictions = []
        aleatorics = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self(x)

                if self.heteroscedastic:
                    # Split output into mean and log variance
                    mean = output[:, :self.output_dim]
                    log_var = output[:, self.output_dim:]
                    aleatoric_var = torch.exp(log_var)
                else:
                    mean = output
                    aleatoric_var = torch.zeros_like(mean)

                predictions.append(mean)
                aleatorics.append(aleatoric_var)

        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, output_dim]
        aleatorics = torch.stack(aleatorics, dim=0)

        # Compute uncertainties
        predictive_mean = predictions.mean(dim=0)
        epistemic_var = predictions.var(dim=0)  # Model uncertainty
        aleatoric_var = aleatorics.mean(dim=0)  # Data uncertainty

        # Standard deviations
        epistemic_std = torch.sqrt(epistemic_var)
        aleatoric_std = torch.sqrt(aleatoric_var)

        return predictive_mean, aleatoric_std, epistemic_std

    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence from all Bayesian layers"""
        kl_div = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLayer):
                kl_div += layer.kl_divergence()
        return kl_div


class BayesianEnsemble(nn.Module):
    """
    Ensemble of Bayesian Neural Networks for enhanced uncertainty quantification.

    Combines multiple BNNs to capture model uncertainty and improve predictions.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_models: int = 5,
                 hidden_layers: List[int] = [100, 100],
                 **bnn_kwargs):
        """
        Initialize Bayesian ensemble.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_models: Number of models in ensemble
            hidden_layers: Hidden layer configuration
            **bnn_kwargs: Additional arguments for BNNs
        """
        super(BayesianEnsemble, self).__init__()

        self.n_models = n_models
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Create ensemble of BNNs
        self.models = nn.ModuleList([
            BayesianNeuralNetwork(
                input_dim, output_dim, hidden_layers, **bnn_kwargs
            ) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor, model_idx: Optional[int] = None) -> torch.Tensor:
        """Forward pass through specific model or all models"""
        if model_idx is not None:
            return self.models[model_idx](x)
        else:
            # Return outputs from all models
            outputs = [model(x) for model in self.models]
            return torch.stack(outputs, dim=0)

    def predict_with_uncertainty(self, x: torch.Tensor,
                                n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Ensemble prediction with comprehensive uncertainty quantification.

        Returns:
            Dictionary with different uncertainty components
        """
        all_predictions = []
        all_aleatorics = []
        all_epistemics = []

        # Get predictions from each model
        for model in self.models:
            mean, aleatoric, epistemic = model.predict_with_uncertainty(x, n_samples)
            all_predictions.append(mean)
            all_aleatorics.append(aleatoric)
            all_epistemics.append(epistemic)

        # Stack ensemble predictions
        ensemble_predictions = torch.stack(all_predictions, dim=0)  # [n_models, batch_size, output_dim]
        ensemble_aleatorics = torch.stack(all_aleatorics, dim=0)
        ensemble_epistemics = torch.stack(all_epistemics, dim=0)

        # Ensemble statistics
        ensemble_mean = ensemble_predictions.mean(dim=0)
        ensemble_var = ensemble_predictions.var(dim=0)

        # Average uncertainties
        avg_aleatoric = ensemble_aleatorics.mean(dim=0)
        avg_epistemic = ensemble_epistemics.mean(dim=0)

        # Total uncertainty (ensemble + within-model)
        total_epistemic = torch.sqrt(ensemble_var + avg_epistemic**2)

        return {
            'mean': ensemble_mean,
            'aleatoric_uncertainty': avg_aleatoric,
            'epistemic_uncertainty': total_epistemic,
            'ensemble_variance': torch.sqrt(ensemble_var),
            'total_uncertainty': torch.sqrt(avg_aleatoric**2 + total_epistemic**2),
            'individual_predictions': ensemble_predictions
        }

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from all models"""
        return sum(model.kl_divergence() for model in self.models)


class BayesianTrainer:
    """
    Trainer for Bayesian Neural Networks with variational inference.

    Implements Evidence Lower Bound (ELBO) optimization with proper
    uncertainty calibration and validation.
    """

    def __init__(self,
                 model: Union[BayesianNeuralNetwork, BayesianEnsemble],
                 learning_rate: float = 1e-3,
                 kl_weight: float = 1.0,
                 optimizer_type: str = 'adam',
                 scheduler_type: Optional[str] = None):
        """
        Initialize Bayesian trainer.

        Args:
            model: Bayesian model to train
            learning_rate: Learning rate
            kl_weight: Weight for KL divergence term
            optimizer_type: Optimizer type
            scheduler_type: Learning rate scheduler type
        """
        self.model = model
        self.kl_weight = kl_weight

        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Setup scheduler
        self.scheduler = None
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

        self.training_history = []

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 1000,
              batch_size: Optional[int] = None,
              kl_annealing: bool = True) -> Dict[str, Any]:
        """
        Train Bayesian Neural Network.

        Args:
            X_train: Training inputs
            y_train: Training outputs
            X_val: Validation inputs
            y_val: Validation outputs
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            kl_annealing: Whether to anneal KL weight

        Returns:
            Training results
        """
        logger.info("Starting Bayesian Neural Network training")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        else:
            X_val_tensor = y_val_tensor = None

        n_batches = 1 if batch_size is None else len(X_train) // batch_size

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_mse = 0
            epoch_kl = 0

            # KL annealing
            if kl_annealing:
                kl_weight = min(1.0, epoch / (epochs * 0.5)) * self.kl_weight
            else:
                kl_weight = self.kl_weight

            # Training loop
            if batch_size is None:
                # Full batch
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(X_train_tensor)

                # Compute losses
                if self.model.heteroscedastic:
                    mse_loss = self._heteroscedastic_loss(output, y_train_tensor)
                else:
                    mse_loss = F.mse_loss(output, y_train_tensor)

                kl_loss = self.model.kl_divergence() / len(X_train)
                total_loss = mse_loss + kl_weight * kl_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_loss = total_loss.item()
                epoch_mse = mse_loss.item()
                epoch_kl = kl_loss.item()

            else:
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train_tensor[i:i+batch_size]
                    batch_y = y_train_tensor[i:i+batch_size]

                    self.optimizer.zero_grad()

                    output = self.model(batch_X)

                    if self.model.heteroscedastic:
                        mse_loss = self._heteroscedastic_loss(output, batch_y)
                    else:
                        mse_loss = F.mse_loss(output, batch_y)

                    kl_loss = self.model.kl_divergence() / len(X_train)
                    total_loss = mse_loss + kl_weight * kl_loss

                    total_loss.backward()
                    self.optimizer.step()

                    epoch_loss += total_loss.item()
                    epoch_mse += mse_loss.item()
                    epoch_kl += kl_loss.item()

                epoch_loss /= n_batches
                epoch_mse /= n_batches
                epoch_kl /= n_batches

            # Validation
            val_loss = None
            if X_val_tensor is not None:
                val_loss = self._validate(X_val_tensor, y_val_tensor)

            # Record history
            epoch_info = {
                'epoch': epoch,
                'train_loss': epoch_loss,
                'train_mse': epoch_mse,
                'train_kl': epoch_kl,
                'kl_weight': kl_weight,
                'val_loss': val_loss
            }
            self.training_history.append(epoch_info)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else epoch_loss)
                else:
                    self.scheduler.step()

            # Logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.6f}, "
                           f"MSE = {epoch_mse:.6f}, KL = {epoch_kl:.6f}")

        logger.info("Bayesian Neural Network training completed")

        return {
            'final_loss': epoch_loss,
            'training_history': self.training_history,
            'model_state': self.model.state_dict()
        }

    def _heteroscedastic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute heteroscedastic loss (negative log likelihood)"""
        mean = output[:, :self.model.output_dim]
        log_var = output[:, self.model.output_dim:]

        # Negative log likelihood
        precision = torch.exp(-log_var)
        loss = 0.5 * (log_var + precision * (target - mean)**2)
        return loss.mean()

    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Compute validation loss"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, BayesianEnsemble):
                # Use first model for validation
                output = self.model.models[0](X_val)
            else:
                output = self.model(X_val)

            if self.model.heteroscedastic:
                val_loss = self._heteroscedastic_loss(output, y_val)
            else:
                val_loss = F.mse_loss(output, y_val)

            return val_loss.item()

    def predict(self, X: np.ndarray,
                n_samples: int = 100,
                return_std: bool = True) -> Dict[str, Any]:
        """
        Make predictions with uncertainty quantification.

        Args:
            X: Input data
            n_samples: Number of Monte Carlo samples
            return_std: Whether to return standard deviations

        Returns:
            Predictions with uncertainty estimates
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        if isinstance(self.model, BayesianEnsemble):
            # Ensemble prediction
            result = self.model.predict_with_uncertainty(X_tensor, n_samples)

            predictions = {
                'mean': result['mean'].numpy(),
                'total_uncertainty': result['total_uncertainty'].numpy(),
                'aleatoric_uncertainty': result['aleatoric_uncertainty'].numpy(),
                'epistemic_uncertainty': result['epistemic_uncertainty'].numpy(),
                'ensemble_variance': result['ensemble_variance'].numpy()
            }
        else:
            # Single model prediction
            mean, aleatoric_std, epistemic_std = self.model.predict_with_uncertainty(X_tensor, n_samples)

            predictions = {
                'mean': mean.numpy(),
                'aleatoric_uncertainty': aleatoric_std.numpy(),
                'epistemic_uncertainty': epistemic_std.numpy(),
                'total_uncertainty': torch.sqrt(aleatoric_std**2 + epistemic_std**2).numpy()
            }

        # Format for API compatibility
        formatted_results = {}
        for i in range(self.model.output_dim):
            output_name = f"output_{i}"

            mean_val = float(predictions['mean'][0, i])
            total_std = float(predictions['total_uncertainty'][0, i])

            formatted_results[output_name] = {
                'prediction': mean_val,
                'uncertainty': {
                    'total_standard_deviation': total_std,
                    'aleatoric_standard_deviation': float(predictions['aleatoric_uncertainty'][0, i]),
                    'epistemic_standard_deviation': float(predictions['epistemic_uncertainty'][0, i]),
                    'variance': total_std**2,
                    'confidence_interval_95': [
                        mean_val - 1.96 * total_std,
                        mean_val + 1.96 * total_std
                    ]
                },
                'bayesian': True
            }

            # Add ensemble-specific info
            if isinstance(self.model, BayesianEnsemble):
                formatted_results[output_name]['uncertainty']['ensemble_variance'] = float(
                    predictions['ensemble_variance'][0, i]
                )

        return formatted_results

    def calibrate_uncertainty(self, X_cal: np.ndarray, y_cal: np.ndarray,
                            n_samples: int = 100) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates using calibration data.

        Returns calibration metrics like reliability and sharpness.
        """
        predictions = self.predict(X_cal, n_samples, return_std=True)

        # Extract predictions and uncertainties
        pred_means = []
        pred_stds = []

        for i in range(self.model.output_dim):
            output_name = f"output_{i}"
            pred_means.append(predictions[output_name]['prediction'])
            pred_stds.append(predictions[output_name]['uncertainty']['total_standard_deviation'])

        pred_means = np.array(pred_means)
        pred_stds = np.array(pred_stds)

        # Compute calibration metrics
        calibration_metrics = self._compute_calibration_metrics(
            y_cal.flatten(), pred_means, pred_stds
        )

        return calibration_metrics

    def _compute_calibration_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_std: np.ndarray) -> Dict[str, float]:
        """Compute uncertainty calibration metrics"""
        # Reliability (calibration error)
        confidence_levels = np.linspace(0.1, 0.9, 9)
        calibration_errors = []

        for conf_level in confidence_levels:
            # Compute prediction intervals
            z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + conf_level) / 2)).item()
            lower = y_pred - z_score * y_std
            upper = y_pred + z_score * y_std

            # Check coverage
            in_interval = (y_true >= lower) & (y_true <= upper)
            actual_coverage = in_interval.mean()

            calibration_error = abs(actual_coverage - conf_level)
            calibration_errors.append(calibration_error)

        # Mean calibration error
        mce = np.mean(calibration_errors)

        # Sharpness (average prediction interval width)
        sharpness = np.mean(2 * 1.96 * y_std)  # 95% interval width

        # Proper scoring metrics
        nll = 0.5 * (np.log(2 * np.pi * y_std**2) + ((y_true - y_pred)**2) / y_std**2)
        mean_nll = np.mean(nll)

        return {
            'mean_calibration_error': float(mce),
            'sharpness': float(sharpness),
            'negative_log_likelihood': float(mean_nll),
            'rmse': float(np.sqrt(np.mean((y_true - y_pred)**2)))
        }


# Factory functions for common configurations
def create_bayesian_surrogate(input_dim: int, output_dim: int,
                             architecture_type: str = 'standard',
                             **kwargs) -> Tuple[BayesianNeuralNetwork, BayesianTrainer]:
    """
    Create Bayesian surrogate model for engineering applications.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        architecture_type: Type of architecture ('standard', 'deep', 'wide')
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, trainer)
    """
    if architecture_type == 'standard':
        hidden_layers = [100, 100]
    elif architecture_type == 'deep':
        hidden_layers = [64, 64, 64, 64]
    elif architecture_type == 'wide':
        hidden_layers = [200, 200]
    else:
        hidden_layers = kwargs.get('hidden_layers', [100, 100])

    model = BayesianNeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        activation=kwargs.get('activation', 'relu'),
        use_mc_dropout=kwargs.get('use_mc_dropout', True),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
        prior_std=kwargs.get('prior_std', 1.0),
        heteroscedastic=kwargs.get('heteroscedastic', True)
    )

    trainer = BayesianTrainer(
        model=model,
        learning_rate=kwargs.get('learning_rate', 1e-3),
        kl_weight=kwargs.get('kl_weight', 1.0),
        optimizer_type=kwargs.get('optimizer_type', 'adam')
    )

    return model, trainer


def create_bayesian_ensemble(input_dim: int, output_dim: int,
                           n_models: int = 5,
                           **kwargs) -> Tuple[BayesianEnsemble, BayesianTrainer]:
    """
    Create Bayesian ensemble for robust uncertainty quantification.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        n_models: Number of models in ensemble
        **kwargs: Additional arguments

    Returns:
        Tuple of (ensemble, trainer)
    """
    ensemble = BayesianEnsemble(
        input_dim=input_dim,
        output_dim=output_dim,
        n_models=n_models,
        hidden_layers=kwargs.get('hidden_layers', [100, 100]),
        activation=kwargs.get('activation', 'relu'),
        use_mc_dropout=kwargs.get('use_mc_dropout', True),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
        prior_std=kwargs.get('prior_std', 1.0),
        heteroscedastic=kwargs.get('heteroscedastic', True)
    )

    trainer = BayesianTrainer(
        model=ensemble,
        learning_rate=kwargs.get('learning_rate', 1e-3),
        kl_weight=kwargs.get('kl_weight', 1.0),
        optimizer_type=kwargs.get('optimizer_type', 'adam')
    )

    return ensemble, trainer