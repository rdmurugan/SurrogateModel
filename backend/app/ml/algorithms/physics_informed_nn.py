import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Callable, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class PhysicsLoss(ABC):
    """Abstract base class for physics-based loss functions"""

    @abstractmethod
    def compute_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                    model: nn.Module) -> torch.Tensor:
        """
        Compute physics-based loss.

        Args:
            model_output: Neural network predictions
            input_coords: Input coordinates
            model: Neural network model for computing derivatives

        Returns:
            Physics loss value
        """
        pass


class ConservationLaw(PhysicsLoss):
    """Conservation law constraint (e.g., mass, energy, momentum conservation)"""

    def __init__(self, law_type: str = 'mass_conservation',
                 coordinates: List[str] = ['x', 'y'],
                 weight: float = 1.0):
        """
        Initialize conservation law.

        Args:
            law_type: Type of conservation law
            coordinates: Coordinate names
            weight: Weight for this physics constraint
        """
        self.law_type = law_type
        self.coordinates = coordinates
        self.weight = weight

    def compute_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                    model: nn.Module) -> torch.Tensor:
        """Compute conservation law loss"""
        if self.law_type == 'mass_conservation':
            return self._mass_conservation_loss(model_output, input_coords, model)
        elif self.law_type == 'energy_conservation':
            return self._energy_conservation_loss(model_output, input_coords, model)
        elif self.law_type == 'momentum_conservation':
            return self._momentum_conservation_loss(model_output, input_coords, model)
        else:
            raise ValueError(f"Unknown conservation law: {self.law_type}")

    def _mass_conservation_loss(self, model_output: torch.Tensor,
                              input_coords: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Mass conservation: ∇·(ρv) + ∂ρ/∂t = 0"""
        # Simplified mass conservation for demonstration
        # Assumes model_output contains [rho, vx, vy] or similar
        batch_size = input_coords.shape[0]

        if model_output.shape[1] < len(self.coordinates) + 1:
            # Not enough outputs for mass conservation
            return torch.tensor(0.0, requires_grad=True)

        # Extract density and velocity components
        rho = model_output[:, 0:1]  # Density
        velocities = model_output[:, 1:len(self.coordinates)+1]  # Velocity components

        # Compute divergence of mass flux
        divergence = torch.zeros(batch_size, 1, requires_grad=True)

        for i, coord_name in enumerate(self.coordinates):
            # Compute derivative of (rho * v_i) with respect to coordinate i
            coord_idx = i  # Assuming coordinates are in order [x, y, z, ...]
            rho_v = rho * velocities[:, i:i+1]

            # Compute derivative using automatic differentiation
            grad_outputs = torch.ones_like(rho_v)
            gradients = torch.autograd.grad(
                outputs=rho_v,
                inputs=input_coords,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]

            divergence = divergence + gradients[:, coord_idx:coord_idx+1]

        # Mass conservation loss
        conservation_loss = torch.mean(divergence**2)
        return self.weight * conservation_loss

    def _energy_conservation_loss(self, model_output: torch.Tensor,
                                input_coords: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Energy conservation constraint"""
        # Placeholder for energy conservation
        # Would implement specific energy conservation equations
        return torch.tensor(0.0, requires_grad=True)

    def _momentum_conservation_loss(self, model_output: torch.Tensor,
                                  input_coords: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Momentum conservation constraint"""
        # Placeholder for momentum conservation
        # Would implement Navier-Stokes or similar equations
        return torch.tensor(0.0, requires_grad=True)


class BoundaryCondition(PhysicsLoss):
    """Boundary condition constraint"""

    def __init__(self, condition_type: str = 'dirichlet',
                 boundary_value: float = 0.0,
                 boundary_function: Optional[Callable] = None,
                 weight: float = 1.0):
        """
        Initialize boundary condition.

        Args:
            condition_type: Type of boundary condition ('dirichlet', 'neumann', 'robin')
            boundary_value: Constant boundary value
            boundary_function: Function defining boundary values
            weight: Weight for this constraint
        """
        self.condition_type = condition_type
        self.boundary_value = boundary_value
        self.boundary_function = boundary_function
        self.weight = weight

    def compute_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                    model: nn.Module) -> torch.Tensor:
        """Compute boundary condition loss"""
        if self.condition_type == 'dirichlet':
            return self._dirichlet_loss(model_output, input_coords)
        elif self.condition_type == 'neumann':
            return self._neumann_loss(model_output, input_coords, model)
        elif self.condition_type == 'robin':
            return self._robin_loss(model_output, input_coords, model)
        else:
            raise ValueError(f"Unknown boundary condition: {self.condition_type}")

    def _dirichlet_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor) -> torch.Tensor:
        """Dirichlet boundary condition: u = g on boundary"""
        if self.boundary_function is not None:
            target_values = self.boundary_function(input_coords)
        else:
            target_values = torch.full_like(model_output[:, 0:1], self.boundary_value)

        boundary_loss = torch.mean((model_output[:, 0:1] - target_values)**2)
        return self.weight * boundary_loss

    def _neumann_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                     model: nn.Module) -> torch.Tensor:
        """Neumann boundary condition: ∂u/∂n = g on boundary"""
        # Compute normal derivative (simplified for demonstration)
        grad_outputs = torch.ones_like(model_output[:, 0:1])
        gradients = torch.autograd.grad(
            outputs=model_output[:, 0:1],
            inputs=input_coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        # Assume normal direction is along first coordinate (simplified)
        normal_derivative = gradients[:, 0:1]

        if self.boundary_function is not None:
            target_derivative = self.boundary_function(input_coords)
        else:
            target_derivative = torch.full_like(normal_derivative, self.boundary_value)

        neumann_loss = torch.mean((normal_derivative - target_derivative)**2)
        return self.weight * neumann_loss

    def _robin_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                   model: nn.Module) -> torch.Tensor:
        """Robin boundary condition: α*u + β*∂u/∂n = g on boundary"""
        # Combined Dirichlet and Neumann (simplified)
        alpha, beta = 1.0, 1.0  # Could be parameters

        dirichlet_part = alpha * model_output[:, 0:1]

        grad_outputs = torch.ones_like(model_output[:, 0:1])
        gradients = torch.autograd.grad(
            outputs=model_output[:, 0:1],
            inputs=input_coords,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]

        neumann_part = beta * gradients[:, 0:1]
        combined = dirichlet_part + neumann_part

        target_value = torch.full_like(combined, self.boundary_value)
        robin_loss = torch.mean((combined - target_value)**2)
        return self.weight * robin_loss


class DimensionalConsistency(PhysicsLoss):
    """Dimensional consistency constraint"""

    def __init__(self, expected_dimensions: Dict[str, str],
                 weight: float = 1.0):
        """
        Initialize dimensional consistency check.

        Args:
            expected_dimensions: Dictionary mapping output names to expected dimensions
            weight: Weight for this constraint
        """
        self.expected_dimensions = expected_dimensions
        self.weight = weight

    def compute_loss(self, model_output: torch.Tensor, input_coords: torch.Tensor,
                    model: nn.Module) -> torch.Tensor:
        """Compute dimensional consistency loss"""
        # Simplified dimensional check
        # In practice, this would be more sophisticated
        consistency_loss = torch.tensor(0.0, requires_grad=True)

        # Check for obviously incorrect dimensional scaling
        for i in range(model_output.shape[1]):
            output_values = model_output[:, i]

            # Simple checks for physical reasonableness
            if torch.any(torch.isnan(output_values)) or torch.any(torch.isinf(output_values)):
                consistency_loss = consistency_loss + 1e6  # High penalty for invalid values

            # Check for extreme values that might indicate dimensional issues
            if torch.max(torch.abs(output_values)) > 1e10:
                consistency_loss = consistency_loss + torch.mean(torch.abs(output_values)) * 1e-6

        return self.weight * consistency_loss


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) implementation.

    This network incorporates physical laws and constraints directly into
    the loss function, enabling better extrapolation and physical consistency.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_layers: List[int] = [50, 50, 50],
                 activation: str = 'tanh',
                 physics_constraints: Optional[List[PhysicsLoss]] = None):
        """
        Initialize Physics-Informed Neural Network.

        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'swish')
            physics_constraints: List of physics constraints
        """
        super(PhysicsInformedNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_constraints = physics_constraints or []

        # Create network layers
        layers = []
        layer_sizes = [input_dim] + hidden_layers + [output_dim]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # Add activation except for output layer
            if i < len(layer_sizes) - 2:
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'swish':
                    layers.append(nn.SiLU())  # SiLU is Swish

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

    def compute_physics_loss(self, input_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute total physics-based loss.

        Args:
            input_coords: Input coordinates (requires_grad=True for derivatives)

        Returns:
            Total physics loss
        """
        # Forward pass
        model_output = self.forward(input_coords)

        total_physics_loss = torch.tensor(0.0, requires_grad=True)

        # Apply all physics constraints
        for constraint in self.physics_constraints:
            physics_loss = constraint.compute_loss(model_output, input_coords, self)
            total_physics_loss = total_physics_loss + physics_loss

        return total_physics_loss

    def add_physics_constraint(self, constraint: PhysicsLoss):
        """Add a new physics constraint"""
        self.physics_constraints.append(constraint)

    def remove_physics_constraint(self, constraint_type: str):
        """Remove physics constraints of a specific type"""
        self.physics_constraints = [
            c for c in self.physics_constraints
            if not isinstance(c, constraint_type)
        ]


class PINNTrainer:
    """Trainer for Physics-Informed Neural Networks"""

    def __init__(self, model: PhysicsInformedNN,
                 data_weight: float = 1.0,
                 physics_weight: float = 1.0,
                 learning_rate: float = 1e-3,
                 optimizer_type: str = 'adam'):
        """
        Initialize PINN trainer.

        Args:
            model: Physics-Informed Neural Network
            data_weight: Weight for data fitting loss
            physics_weight: Weight for physics loss
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'lbfgs')
        """
        self.model = model
        self.data_weight = data_weight
        self.physics_weight = physics_weight

        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            self.optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.training_history = []

    def train(self, X_data: np.ndarray, y_data: np.ndarray,
              X_physics: np.ndarray,
              epochs: int = 1000,
              batch_size: Optional[int] = None,
              validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the Physics-Informed Neural Network.

        Args:
            X_data: Training input data
            y_data: Training output data
            X_physics: Physics constraint points
            epochs: Number of training epochs
            batch_size: Batch size (None for full batch)
            validation_split: Fraction of data for validation

        Returns:
            Training results
        """
        logger.info("Starting PINN training")

        # Convert to tensors
        X_data_tensor = torch.FloatTensor(X_data)
        y_data_tensor = torch.FloatTensor(y_data)
        X_physics_tensor = torch.FloatTensor(X_physics)
        X_physics_tensor.requires_grad_(True)

        # Split data for validation
        n_val = int(len(X_data) * validation_split)
        if n_val > 0:
            X_val = X_data_tensor[-n_val:]
            y_val = y_data_tensor[-n_val:]
            X_train = X_data_tensor[:-n_val]
            y_train = y_data_tensor[:-n_val]
        else:
            X_train, y_train = X_data_tensor, y_data_tensor
            X_val, y_val = None, None

        # Training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Data fitting loss
            y_pred = self.model(X_train)
            data_loss = nn.MSELoss()(y_pred, y_train)

            # Physics loss
            physics_loss = self.model.compute_physics_loss(X_physics_tensor)

            # Total loss
            total_loss = (self.data_weight * data_loss +
                         self.physics_weight * physics_loss)

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Validation loss
            val_loss = None
            if X_val is not None:
                with torch.no_grad():
                    y_val_pred = self.model(X_val)
                    val_loss = nn.MSELoss()(y_val_pred, y_val)

            # Record training history
            epoch_info = {
                'epoch': epoch,
                'total_loss': float(total_loss.item()),
                'data_loss': float(data_loss.item()),
                'physics_loss': float(physics_loss.item()),
                'validation_loss': float(val_loss.item()) if val_loss is not None else None
            }
            self.training_history.append(epoch_info)

            # Logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                           f"Data Loss = {data_loss.item():.6f}, "
                           f"Physics Loss = {physics_loss.item():.6f}")

        logger.info("PINN training completed")

        return {
            'final_loss': float(total_loss.item()),
            'training_history': self.training_history,
            'model_state': self.model.state_dict()
        }

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions with the trained PINN.

        Args:
            X: Input points

        Returns:
            Predictions with uncertainty estimates
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred = self.model(X_tensor)
            predictions = y_pred.numpy()

        # Simple uncertainty estimation (could be improved)
        # For now, use a constant uncertainty
        uncertainty = np.ones_like(predictions) * 0.1

        results = {}
        for i in range(predictions.shape[1]):
            output_name = f"output_{i}"
            pred_val = float(predictions[0, i])
            std_val = float(uncertainty[0, i])

            results[output_name] = {
                'prediction': pred_val,
                'uncertainty': {
                    'standard_deviation': std_val,
                    'variance': std_val**2,
                    'confidence_interval_95': [
                        pred_val - 1.96 * std_val,
                        pred_val + 1.96 * std_val
                    ]
                },
                'physics_informed': True
            }

        return results

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics"""
        if not self.training_history:
            return {}

        history_array = np.array([[h['data_loss'], h['physics_loss'], h['total_loss']]
                                 for h in self.training_history])

        metrics = {
            'final_data_loss': float(history_array[-1, 0]),
            'final_physics_loss': float(history_array[-1, 1]),
            'final_total_loss': float(history_array[-1, 2]),
            'loss_reduction': {
                'data_loss': float(history_array[0, 0] - history_array[-1, 0]),
                'physics_loss': float(history_array[0, 1] - history_array[-1, 1]),
                'total_loss': float(history_array[0, 2] - history_array[-1, 2])
            },
            'convergence_epoch': self._find_convergence_epoch(),
            'physics_vs_data_ratio': float(history_array[-1, 1] / (history_array[-1, 0] + 1e-8))
        }

        return metrics

    def _find_convergence_epoch(self, tolerance: float = 1e-4) -> int:
        """Find epoch where training converged"""
        if len(self.training_history) < 10:
            return len(self.training_history)

        losses = [h['total_loss'] for h in self.training_history]

        # Look for plateau in loss
        for i in range(10, len(losses)):
            recent_losses = losses[i-10:i]
            if np.std(recent_losses) < tolerance:
                return i

        return len(self.training_history)


# Factory function for creating common PINN configurations
def create_engineering_pinn(problem_type: str, input_dim: int, output_dim: int,
                          **kwargs) -> Tuple[PhysicsInformedNN, List[PhysicsLoss]]:
    """
    Create PINN with common engineering physics constraints.

    Args:
        problem_type: Type of engineering problem ('fluid_flow', 'heat_transfer', 'structural')
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional arguments

    Returns:
        Tuple of (PINN model, physics constraints list)
    """
    constraints = []

    if problem_type == 'fluid_flow':
        # Add mass and momentum conservation
        constraints.append(ConservationLaw('mass_conservation', weight=1.0))
        constraints.append(ConservationLaw('momentum_conservation', weight=0.5))

        # Add boundary conditions
        constraints.append(BoundaryCondition('dirichlet', boundary_value=0.0, weight=1.0))

    elif problem_type == 'heat_transfer':
        # Add energy conservation
        constraints.append(ConservationLaw('energy_conservation', weight=1.0))

        # Add thermal boundary conditions
        constraints.append(BoundaryCondition('dirichlet', boundary_value=300.0, weight=1.0))
        constraints.append(BoundaryCondition('neumann', boundary_value=0.0, weight=0.5))

    elif problem_type == 'structural':
        # Add equilibrium conditions
        constraints.append(ConservationLaw('momentum_conservation', weight=1.0))

        # Add displacement boundary conditions
        constraints.append(BoundaryCondition('dirichlet', boundary_value=0.0, weight=1.0))

    # Always add dimensional consistency
    constraints.append(DimensionalConsistency({}, weight=0.1))

    # Create PINN model
    model = PhysicsInformedNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=kwargs.get('hidden_layers', [50, 50, 50]),
        activation=kwargs.get('activation', 'tanh'),
        physics_constraints=constraints
    )

    return model, constraints