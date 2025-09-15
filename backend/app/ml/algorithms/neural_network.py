import numpy as np
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ..base import SurrogateModelBase


class NeuralNetworkSurrogate(SurrogateModelBase):
    """
    Neural Network surrogate model using PyTorch.

    Excellent for:
    - Large datasets (> 1,000 samples)
    - Complex, nonlinear relationships
    - High-dimensional problems
    - When accuracy is more important than interpretability

    Advantages:
    - Can approximate any continuous function
    - Handles high-dimensional inputs well
    - Fast prediction once trained
    - Can learn complex patterns

    Disadvantages:
    - Requires more data than other methods
    - Black box (less interpretable)
    - Sensitive to hyperparameters
    - May overfit without proper regularization
    """

    def __init__(self, **hyperparameters):
        default_params = {
            'hidden_layers': [64, 32],
            'activation': 'relu',
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 200,
            'validation_split': 0.2,
            'early_stopping_patience': 20,
            'l2_regularization': 0.01,
            'uncertainty_estimation': 'dropout'  # 'dropout', 'ensemble', 'none'
        }
        default_params.update(hyperparameters)
        super().__init__(**default_params)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {}

    def _create_model(self) -> nn.Module:
        """Create neural network architecture"""
        hidden_layers = self.hyperparameters.get('hidden_layers', [64, 32])
        activation = self.hyperparameters.get('activation', 'relu')
        dropout_rate = self.hyperparameters.get('dropout_rate', 0.1)

        # Determine input/output dimensions
        input_dim = len(self.feature_names) if self.feature_names else 1
        output_dim = len(self.target_names) if self.target_names else 1

        return NeuralNetworkModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate
        ).to(self.device)

    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the neural network"""
        # Split data for validation
        validation_split = self.hyperparameters.get('validation_split', 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        batch_size = self.hyperparameters.get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer and loss function
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        l2_reg = self.hyperparameters.get('l2_regularization', 0.01)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        criterion = nn.MSELoss()

        # Training loop
        epochs = self.hyperparameters.get('epochs', 200)
        patience = self.hyperparameters.get('early_stopping_patience', 20)

        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                # Restore best model state
                self.model.load_state_dict(self.best_model_state)
                break

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the neural network"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()

    def get_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty estimates using Monte Carlo dropout or ensemble"""
        uncertainty_method = self.hyperparameters.get('uncertainty_estimation', 'dropout')

        if uncertainty_method == 'dropout':
            return self._get_dropout_uncertainty(X)
        elif uncertainty_method == 'ensemble':
            return self._get_ensemble_uncertainty(X)
        else:
            # Return dummy uncertainty
            predictions = self._predict_model(X)
            uncertainty = {}
            for i, target_name in enumerate(self.target_names):
                pred_val = float(predictions[0, i]) if predictions.ndim > 1 else float(predictions[0])
                std_val = abs(pred_val) * 0.05  # 5% of prediction as uncertainty

                uncertainty[target_name] = {
                    'standard_deviation': std_val,
                    'variance': std_val**2,
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ]
                }
            return uncertainty

    def _get_dropout_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """Get uncertainty using Monte Carlo dropout"""
        self.model.train()  # Enable dropout
        X_tensor = torch.FloatTensor(X).to(self.device)

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)  # Shape: (n_samples, n_points, n_outputs)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

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
                ]
            }

        return uncertainty

    def _get_ensemble_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get uncertainty using ensemble of models (placeholder)"""
        # This would require training multiple models
        # For now, return dropout uncertainty
        return self._get_dropout_uncertainty(X)

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for monitoring convergence"""
        return self.training_history

    @staticmethod
    def get_default_hyperparameters() -> Dict[str, Any]:
        """Get default hyperparameters for optimization"""
        return {
            'hidden_layers': [[32], [64], [64, 32], [128, 64], [128, 64, 32]],
            'activation': ['relu', 'tanh', 'leaky_relu'],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'l2_regularization': [0.0, 0.001, 0.01, 0.1]
        }


class NeuralNetworkModel(nn.Module):
    """PyTorch neural network model"""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int],
                 activation: str = 'relu', dropout_rate: float = 0.1):
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        return self.network(x)