import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from typing import Dict, Any, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    Adds position information to sequence embeddings using sinusoidal patterns.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    """
    Feature-level attention mechanism for identifying important features.

    Computes attention weights for different input features to understand
    which parameters are most important for predictions.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super(FeatureAttention, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Attention computation
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Feature transformation
        self.feature_transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply feature attention.

        Args:
            x: Input features [batch_size, seq_len, feature_dim]

        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, seq_len, feature_dim = x.shape

        # Reshape for attention computation
        x_flat = x.view(-1, feature_dim)  # [batch_size * seq_len, feature_dim]

        # Compute attention scores
        attention_scores = self.attention_net(x_flat)  # [batch_size * seq_len, 1]
        attention_scores = attention_scores.view(batch_size, seq_len, 1)

        # Apply softmax across sequence dimension
        attention_weights = F.softmax(attention_scores, dim=1)

        # Transform features
        transformed_features = self.feature_transform(x)

        # Apply attention
        attended_features = transformed_features * attention_weights

        return attended_features, attention_weights


class MultiModalFusion(nn.Module):
    """
    Multi-modal data fusion for combining different types of inputs.

    Handles fusion of tabular data, time series, and other modalities
    for comprehensive surrogate modeling.
    """

    def __init__(self,
                 modal_dims: Dict[str, int],
                 fusion_dim: int = 128,
                 fusion_type: str = 'attention'):
        """
        Initialize multi-modal fusion.

        Args:
            modal_dims: Dictionary of {modality_name: feature_dimension}
            fusion_dim: Dimension for fusion representations
            fusion_type: Type of fusion ('attention', 'concat', 'gated')
        """
        super(MultiModalFusion, self).__init__()

        self.modal_dims = modal_dims
        self.fusion_dim = fusion_dim
        self.fusion_type = fusion_type

        # Modal encoders
        self.modal_encoders = nn.ModuleDict()
        for modality, dim in modal_dims.items():
            self.modal_encoders[modality] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.BatchNorm1d(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim)
            )

        # Fusion mechanisms
        if fusion_type == 'attention':
            self.attention_weights = nn.Linear(fusion_dim, 1)
        elif fusion_type == 'gated':
            self.gate_net = nn.Sequential(
                nn.Linear(fusion_dim * len(modal_dims), fusion_dim),
                nn.Sigmoid()
            )

    def forward(self, modal_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-modal inputs.

        Args:
            modal_inputs: Dictionary of {modality_name: tensor}

        Returns:
            Fused representation
        """
        # Encode each modality
        encoded_modals = {}
        for modality, input_tensor in modal_inputs.items():
            if modality in self.modal_encoders:
                encoded = self.modal_encoders[modality](input_tensor)
                encoded_modals[modality] = encoded

        # Fusion
        if self.fusion_type == 'concat':
            # Simple concatenation
            fused = torch.cat(list(encoded_modals.values()), dim=-1)

        elif self.fusion_type == 'attention':
            # Attention-based fusion
            modalities = list(encoded_modals.values())
            stacked = torch.stack(modalities, dim=1)  # [batch, n_modalities, fusion_dim]

            # Compute attention weights
            attention_scores = self.attention_weights(stacked)  # [batch, n_modalities, 1]
            attention_weights = F.softmax(attention_scores, dim=1)

            # Weighted combination
            fused = (stacked * attention_weights).sum(dim=1)

        elif self.fusion_type == 'gated':
            # Gated fusion
            concatenated = torch.cat(list(encoded_modals.values()), dim=-1)
            gate = self.gate_net(concatenated)

            # Apply gating to mean of modalities
            mean_modal = torch.stack(list(encoded_modals.values()), dim=1).mean(dim=1)
            fused = gate * mean_modal

        else:
            # Default: element-wise mean
            fused = torch.stack(list(encoded_modals.values()), dim=1).mean(dim=1)

        return fused


class OptimizationTransformer(nn.Module):
    """
    Transformer model for sequential optimization problems.

    Designed for optimization sequences where each step depends on
    previous evaluations and decisions.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 use_feature_attention: bool = True):
        """
        Initialize Optimization Transformer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            use_feature_attention: Whether to use feature attention
        """
        super(OptimizationTransformer, self).__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_feature_attention = use_feature_attention

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Feature attention
        if use_feature_attention:
            self.feature_attention = FeatureAttention(input_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )

        # Uncertainty estimation (for Bayesian-like behavior)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, output_dim),
            nn.Softplus()  # Ensure positive variance
        )

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through optimization transformer.

        Args:
            src: Source sequence [batch_size, src_seq_len, input_dim]
            tgt: Target sequence [batch_size, tgt_seq_len, input_dim] (for training)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optional attention weights
        """
        batch_size, seq_len, _ = src.shape

        # Feature attention
        attention_weights = None
        if self.use_feature_attention:
            src, attention_weights = self.feature_attention(src)

        # Input embedding
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)

        # Encoder
        memory = self.transformer_encoder(src_embedded)

        # Decoder (for autoregressive prediction)
        if tgt is not None:
            # Training mode with target sequence
            tgt_embedded = self.input_embedding(tgt) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded)

            decoder_output = self.transformer_decoder(tgt_embedded, memory)
        else:
            # Inference mode - use encoder output
            decoder_output = memory

        # Output projections
        predictions = self.output_projection(decoder_output)
        uncertainties = self.uncertainty_head(decoder_output)

        result = {
            'predictions': predictions,
            'uncertainties': uncertainties
        }

        if return_attention and attention_weights is not None:
            result['feature_attention'] = attention_weights

        return result

    def generate_sequence(self, src: torch.Tensor, max_length: int = 50,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Generate optimization sequence autoregressively.

        Args:
            src: Initial sequence
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated sequence
        """
        self.eval()
        generated = src.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions for current sequence
                output = self.forward(generated)
                predictions = output['predictions']

                # Get next step prediction
                next_pred = predictions[:, -1:, :]  # Last time step

                # Apply temperature
                if temperature != 1.0:
                    next_pred = next_pred / temperature

                # Append to sequence
                generated = torch.cat([generated, next_pred], dim=1)

        return generated


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time-series surrogate modeling.

    Specialized for temporal data with trends, seasonality,
    and irregular sampling patterns.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 forecast_horizon: int = 1):
        """
        Initialize Time Series Transformer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            forecast_horizon: Number of steps to forecast
        """
        super(TimeSeriesTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Time encoding (learnable)
        self.time_embedding = nn.Embedding(10000, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Temporal attention for trend/seasonality
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(d_model, output_dim) for _ in range(forecast_horizon)
        ])

        # Uncertainty estimation
        self.uncertainty_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, output_dim),
                nn.Softplus()
            ) for _ in range(forecast_horizon)
        ])

    def forward(self, x: torch.Tensor, time_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for time series prediction.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            time_indices: Time indices for each step [batch_size, seq_len]

        Returns:
            Predictions with uncertainty for each forecast step
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add time encoding
        if time_indices is not None:
            time_emb = self.time_embedding(time_indices)
            x = x + time_emb
        else:
            # Default sequential time indices
            time_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            time_emb = self.time_embedding(time_indices)
            x = x + time_emb

        # Transformer encoding
        encoded = self.transformer(x)

        # Temporal attention for capturing long-range dependencies
        attended, attention_weights = self.temporal_attention(encoded, encoded, encoded)
        combined = encoded + attended

        # Multi-step forecasting
        predictions = []
        uncertainties = []

        # Use last encoded state for forecasting
        last_state = combined[:, -1, :]  # [batch_size, d_model]

        for i in range(self.forecast_horizon):
            pred = self.output_layers[i](last_state)
            uncertainty = self.uncertainty_layers[i](last_state)

            predictions.append(pred)
            uncertainties.append(uncertainty)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch_size, forecast_horizon, output_dim]
        uncertainties = torch.stack(uncertainties, dim=1)

        return {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'attention_weights': attention_weights,
            'encoded_states': combined
        }


class TransferLearningTransformer(nn.Module):
    """
    Transformer with transfer learning capabilities.

    Pre-trained on multiple similar problems and fine-tuned
    for specific applications.
    """

    def __init__(self,
                 base_transformer: nn.Module,
                 target_input_dim: int,
                 target_output_dim: int,
                 adaptation_layers: int = 2,
                 freeze_base: bool = False):
        """
        Initialize transfer learning transformer.

        Args:
            base_transformer: Pre-trained transformer model
            target_input_dim: Target task input dimension
            target_output_dim: Target task output dimension
            adaptation_layers: Number of adaptation layers
            freeze_base: Whether to freeze base model parameters
        """
        super(TransferLearningTransformer, self).__init__()

        self.base_transformer = base_transformer
        self.freeze_base = freeze_base

        if freeze_base:
            # Freeze base transformer parameters
            for param in base_transformer.parameters():
                param.requires_grad = False

        # Domain adaptation layers
        base_dim = base_transformer.d_model
        adaptation_dims = [base_dim] + [base_dim // (2**i) for i in range(adaptation_layers)] + [target_output_dim]

        self.adaptation_layers = nn.ModuleList()
        for i in range(len(adaptation_dims) - 1):
            self.adaptation_layers.append(
                nn.Sequential(
                    nn.Linear(adaptation_dims[i], adaptation_dims[i+1]),
                    nn.ReLU() if i < len(adaptation_dims) - 2 else nn.Identity(),
                    nn.Dropout(0.1) if i < len(adaptation_dims) - 2 else nn.Identity()
                )
            )

        # Input adaptation
        self.input_adapter = nn.Linear(target_input_dim, base_transformer.input_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with domain adaptation"""
        # Adapt input to base model format
        adapted_input = self.input_adapter(x)

        # Base transformer forward pass
        if self.freeze_base:
            with torch.no_grad():
                base_output = self.base_transformer(adapted_input)
        else:
            base_output = self.base_transformer(adapted_input)

        # Extract features (use last layer representation)
        if isinstance(base_output, dict):
            features = base_output.get('encoded_states', base_output['predictions'])
        else:
            features = base_output

        # Domain adaptation
        adapted_features = features
        for layer in self.adaptation_layers:
            adapted_features = layer(adapted_features)

        return {
            'predictions': adapted_features,
            'base_features': features
        }


class TransformerTrainer:
    """
    Trainer for transformer-based surrogate models.

    Handles training with attention visualization, transfer learning,
    and multi-task optimization.
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine'):
        """
        Initialize transformer trainer.

        Args:
            model: Transformer model to train
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_type: Learning rate scheduler type
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=10, factor=0.5
            )
        else:
            self.scheduler = None

        self.training_history = []

    def train(self,
              train_data: torch.utils.data.DataLoader,
              val_data: Optional[torch.utils.data.DataLoader] = None,
              epochs: int = 100,
              loss_type: str = 'mse') -> Dict[str, Any]:
        """
        Train transformer model.

        Args:
            train_data: Training data loader
            val_data: Validation data loader
            epochs: Number of training epochs
            loss_type: Loss function type ('mse', 'huber', 'quantile')

        Returns:
            Training results
        """
        logger.info("Starting Transformer training")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0

            for batch in train_data:
                self.optimizer.zero_grad()

                # Forward pass
                if len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                else:
                    # Handle different batch formats
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None
                    outputs = self.model(inputs)

                # Compute loss
                if isinstance(outputs, dict):
                    predictions = outputs['predictions']
                else:
                    predictions = outputs

                if loss_type == 'mse':
                    loss = F.mse_loss(predictions, targets)
                elif loss_type == 'huber':
                    loss = F.huber_loss(predictions, targets)
                else:
                    loss = F.mse_loss(predictions, targets)

                # Add uncertainty loss if available
                if isinstance(outputs, dict) and 'uncertainties' in outputs:
                    uncertainties = outputs['uncertainties']
                    # Negative log-likelihood loss
                    nll_loss = 0.5 * (torch.log(uncertainties) + (predictions - targets)**2 / uncertainties)
                    loss = loss + 0.1 * nll_loss.mean()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation phase
            val_loss = 0.0
            if val_data is not None:
                val_loss = self._validate(val_data, loss_type)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_data else train_loss)
                else:
                    self.scheduler.step()

            # Record history
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_info)

            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                           f"Val Loss = {val_loss:.6f}")

        logger.info("Transformer training completed")

        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'training_history': self.training_history
        }

    def _validate(self, val_data: torch.utils.data.DataLoader, loss_type: str) -> float:
        """Compute validation loss"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_data:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None

                outputs = self.model(inputs)

                if isinstance(outputs, dict):
                    predictions = outputs['predictions']
                else:
                    predictions = outputs

                if loss_type == 'mse':
                    loss = F.mse_loss(predictions, targets)
                elif loss_type == 'huber':
                    loss = F.huber_loss(predictions, targets)
                else:
                    loss = F.mse_loss(predictions, targets)

                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def predict_with_attention(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions and return attention weights"""
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                outputs = self.model(inputs, return_attention=True)
            else:
                outputs = self.model(inputs)

        return outputs


# Factory functions
def create_optimization_transformer(input_dim: int, output_dim: int,
                                  architecture: str = 'standard') -> OptimizationTransformer:
    """Create transformer for optimization sequences"""
    if architecture == 'standard':
        config = {
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4
        }
    elif architecture == 'large':
        config = {
            'd_model': 512,
            'nhead': 16,
            'num_encoder_layers': 8,
            'num_decoder_layers': 8
        }
    else:
        config = {
            'd_model': 128,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2
        }

    return OptimizationTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )


def create_timeseries_transformer(input_dim: int, output_dim: int = 1,
                                forecast_horizon: int = 1) -> TimeSeriesTransformer:
    """Create transformer for time series prediction"""
    return TimeSeriesTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon
    )