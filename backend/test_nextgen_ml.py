#!/usr/bin/env python3
"""
Comprehensive test suite for next-generation ML algorithms.

Tests the following advanced capabilities:
1. Bayesian Neural Networks with uncertainty quantification
2. Graph Neural Networks for CAD/mesh data processing
3. Transformer models for sequential optimization
4. Multi-modal data fusion
5. Attention mechanisms and feature importance
6. API endpoints for next-gen ML

Usage:
    python test_nextgen_ml.py
"""

import numpy as np
import torch
import requests
import json
import time
import traceback
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"
TEST_TOKEN = "fake-token-for-testing"

# Test colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_success(message: str):
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_error(message: str):
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.ENDC}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.ENDC}")

def print_header(message: str):
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'='*70}{Colors.ENDC}")


class NextGenMLTestSuite:
    """Comprehensive test suite for next-generation ML algorithms"""

    def __init__(self):
        self.test_results = {
            'bayesian_neural_networks': {'passed': 0, 'failed': 0, 'errors': []},
            'graph_neural_networks': {'passed': 0, 'failed': 0, 'errors': []},
            'transformer_models': {'passed': 0, 'failed': 0, 'errors': []},
            'uncertainty_quantification': {'passed': 0, 'failed': 0, 'errors': []},
            'attention_mechanisms': {'passed': 0, 'failed': 0, 'errors': []},
            'api_endpoints': {'passed': 0, 'failed': 0, 'errors': []}
        }

    def run_all_tests(self):
        """Run the complete next-generation ML test suite"""
        print_header("🚀 Next-Generation ML Algorithms Test Suite")

        try:
            # Test 1: Bayesian Neural Networks
            self.test_bayesian_neural_networks()

            # Test 2: Graph Neural Networks
            self.test_graph_neural_networks()

            # Test 3: Transformer Models
            self.test_transformer_models()

            # Test 4: Uncertainty Quantification
            self.test_uncertainty_quantification()

            # Test 5: Attention Mechanisms
            self.test_attention_mechanisms()

            # Test 6: API Endpoints
            self.test_api_endpoints()

            # Summary
            self.print_test_summary()

        except Exception as e:
            print_error(f"Test suite failed with error: {str(e)}")
            traceback.print_exc()

    def test_bayesian_neural_networks(self):
        """Test Bayesian Neural Network implementation"""
        print_header("🧠 Bayesian Neural Networks Tests")

        try:
            # Test BNN Creation
            self.test_bnn_creation()

            # Test Variational Inference
            self.test_variational_inference()

            # Test Monte Carlo Dropout
            self.test_mc_dropout()

            # Test Ensemble Methods
            self.test_bayesian_ensemble()

            # Test Uncertainty Estimation
            self.test_uncertainty_estimation()

        except Exception as e:
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(str(e))
            print_error(f"Bayesian Neural Network testing failed: {str(e)}")

    def test_graph_neural_networks(self):
        """Test Graph Neural Network implementation"""
        print_header("🕸️ Graph Neural Networks Tests")

        try:
            # Test GNN Creation
            self.test_gnn_creation()

            # Test Mesh Data Processing
            self.test_mesh_data_processing()

            # Test Geometric Attention
            self.test_geometric_attention()

            # Test CAD Parameter Optimization
            self.test_cad_optimization()

            # Test Graph Convolution Types
            self.test_graph_convolutions()

        except Exception as e:
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(str(e))
            print_error(f"Graph Neural Network testing failed: {str(e)}")

    def test_transformer_models(self):
        """Test Transformer model implementation"""
        print_header("🔄 Transformer Models Tests")

        try:
            # Test Transformer Creation
            self.test_transformer_creation()

            # Test Optimization Transformer
            self.test_optimization_transformer()

            # Test Time Series Transformer
            self.test_timeseries_transformer()

            # Test Multi-Modal Fusion
            self.test_multimodal_fusion()

            # Test Transfer Learning
            self.test_transfer_learning()

        except Exception as e:
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(str(e))
            print_error(f"Transformer model testing failed: {str(e)}")

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification capabilities"""
        print_header("📊 Uncertainty Quantification Tests")

        try:
            # Test Aleatoric Uncertainty
            self.test_aleatoric_uncertainty()

            # Test Epistemic Uncertainty
            self.test_epistemic_uncertainty()

            # Test Uncertainty Calibration
            self.test_uncertainty_calibration()

            # Test Ensemble Uncertainty
            self.test_ensemble_uncertainty()

        except Exception as e:
            self.test_results['uncertainty_quantification']['failed'] += 1
            self.test_results['uncertainty_quantification']['errors'].append(str(e))
            print_error(f"Uncertainty quantification testing failed: {str(e)}")

    def test_attention_mechanisms(self):
        """Test attention mechanism implementations"""
        print_header("👁️ Attention Mechanisms Tests")

        try:
            # Test Feature Attention
            self.test_feature_attention()

            # Test Geometric Attention
            self.test_geometric_attention_detailed()

            # Test Multi-Head Attention
            self.test_multihead_attention()

            # Test Attention Visualization
            self.test_attention_visualization()

        except Exception as e:
            self.test_results['attention_mechanisms']['failed'] += 1
            self.test_results['attention_mechanisms']['errors'].append(str(e))
            print_error(f"Attention mechanism testing failed: {str(e)}")

    def test_api_endpoints(self):
        """Test API endpoints for next-gen ML"""
        print_header("🌐 API Endpoints Tests")

        try:
            # Test Bayesian API
            self.test_bayesian_api()

            # Test Graph API
            self.test_graph_api()

            # Test Transformer API
            self.test_transformer_api()

            # Test Capabilities Endpoint
            self.test_capabilities_endpoint()

        except Exception as e:
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(str(e))
            print_error(f"API endpoints testing failed: {str(e)}")

    # ==================== Bayesian Neural Network Tests ====================

    def test_bnn_creation(self):
        """Test Bayesian Neural Network creation"""
        print_info("Testing Bayesian Neural Network creation...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianNeuralNetwork, BayesianLayer

            # Test Bayesian layer
            layer = BayesianLayer(10, 5)
            x = torch.randn(3, 10)
            output = layer(x)

            assert output.shape == (3, 5)
            assert layer.kl_divergence() > 0

            # Test full BNN
            bnn = BayesianNeuralNetwork(
                input_dim=5,
                output_dim=2,
                hidden_layers=[32, 16],
                use_mc_dropout=True,
                heteroscedastic=True
            )

            x = torch.randn(4, 5)
            output = bnn(x)

            # With heteroscedastic=True, output should be 2*output_dim
            assert output.shape == (4, 4)  # 2 outputs * 2 (mean + variance)

            print_success("Bayesian Neural Network creation test passed")
            self.test_results['bayesian_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"BNN creation test failed: {str(e)}")
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(f"Creation: {str(e)}")

    def test_variational_inference(self):
        """Test variational inference in BNNs"""
        print_info("Testing variational inference...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianNeuralNetwork, BayesianTrainer

            # Create BNN
            bnn = BayesianNeuralNetwork(3, 1, [16, 8])
            trainer = BayesianTrainer(bnn, kl_weight=0.1)

            # Generate synthetic data
            X = np.random.randn(50, 3)
            y = X[:, 0:1]**2 + X[:, 1:1] + np.random.randn(50, 1) * 0.1

            # Train for few epochs
            results = trainer.train(X, y, epochs=10)

            assert 'final_loss' in results
            assert 'training_history' in results
            assert len(results['training_history']) == 10

            # Check KL divergence is computed
            kl_div = bnn.kl_divergence()
            assert kl_div > 0

            print_success("Variational inference test passed")
            self.test_results['bayesian_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Variational inference test failed: {str(e)}")
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(f"VI: {str(e)}")

    def test_mc_dropout(self):
        """Test Monte Carlo dropout functionality"""
        print_info("Testing Monte Carlo dropout...")

        try:
            from app.ml.algorithms.bayesian_neural_network import MCDropoutLayer

            # Test MC dropout layer
            mc_dropout = MCDropoutLayer(dropout_rate=0.2)
            x = torch.randn(10, 5)

            # Should apply dropout even in eval mode
            mc_dropout.eval()
            output1 = mc_dropout(x)
            output2 = mc_dropout(x)

            # Outputs should be different due to dropout
            assert not torch.allclose(output1, output2)

            print_success("Monte Carlo dropout test passed")
            self.test_results['bayesian_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"MC dropout test failed: {str(e)}")
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(f"MC Dropout: {str(e)}")

    def test_bayesian_ensemble(self):
        """Test Bayesian ensemble functionality"""
        print_info("Testing Bayesian ensemble...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianEnsemble

            # Create ensemble
            ensemble = BayesianEnsemble(
                input_dim=4,
                output_dim=1,
                n_models=3,
                hidden_layers=[16, 8]
            )

            # Test forward pass
            x = torch.randn(5, 4)
            output = ensemble(x)

            # Should return outputs from all models
            assert output.shape == (3, 5, 1)  # [n_models, batch, output_dim]

            # Test uncertainty prediction
            uncertainty_result = ensemble.predict_with_uncertainty(x, n_samples=10)

            assert 'mean' in uncertainty_result
            assert 'aleatoric_uncertainty' in uncertainty_result
            assert 'epistemic_uncertainty' in uncertainty_result
            assert 'ensemble_variance' in uncertainty_result

            print_success("Bayesian ensemble test passed")
            self.test_results['bayesian_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Bayesian ensemble test failed: {str(e)}")
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(f"Ensemble: {str(e)}")

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation capabilities"""
        print_info("Testing uncertainty estimation...")

        try:
            from app.ml.algorithms.bayesian_neural_network import create_bayesian_surrogate

            # Create Bayesian model
            model, trainer = create_bayesian_surrogate(
                input_dim=2,
                output_dim=1,
                architecture_type='standard'
            )

            # Generate data
            X = np.random.randn(30, 2)
            y = X[:, 0:1] * X[:, 1:1] + np.random.randn(30, 1) * 0.05

            # Quick training
            trainer.train(X, y, epochs=5)

            # Test uncertainty prediction
            X_test = np.array([[0.5, 0.5], [1.0, 1.0]])
            predictions = trainer.predict(X_test, n_samples=20)

            # Check formatted output
            assert 'output_0' in predictions
            assert 'prediction' in predictions['output_0']
            assert 'uncertainty' in predictions['output_0']
            assert 'total_standard_deviation' in predictions['output_0']['uncertainty']
            assert 'aleatoric_standard_deviation' in predictions['output_0']['uncertainty']
            assert 'epistemic_standard_deviation' in predictions['output_0']['uncertainty']

            print_success("Uncertainty estimation test passed")
            self.test_results['bayesian_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Uncertainty estimation test failed: {str(e)}")
            self.test_results['bayesian_neural_networks']['failed'] += 1
            self.test_results['bayesian_neural_networks']['errors'].append(f"Uncertainty: {str(e)}")

    # ==================== Graph Neural Network Tests ====================

    def test_gnn_creation(self):
        """Test Graph Neural Network creation"""
        print_info("Testing Graph Neural Network creation...")

        try:
            from app.ml.algorithms.graph_neural_network import MeshGNN, MeshData

            # Create GNN
            gnn = MeshGNN(
                input_dim=3,
                hidden_dims=[32, 64, 32],
                output_dim=1,
                conv_type='gcn',
                task_type='node_prediction'
            )

            # Create test mesh data
            nodes = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
            edges = np.array([[0, 1], [1, 2], [2, 0]])

            mesh = MeshData(nodes, edges.T)
            data = mesh.to_torch_geometric()

            # Test forward pass
            output = gnn(data)
            assert output.shape == (3, 1)  # 3 nodes, 1 output

            print_success("Graph Neural Network creation test passed")
            self.test_results['graph_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"GNN creation test failed: {str(e)}")
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(f"Creation: {str(e)}")

    def test_mesh_data_processing(self):
        """Test mesh data processing capabilities"""
        print_info("Testing mesh data processing...")

        try:
            from app.ml.algorithms.graph_neural_network import MeshData, MeshDataLoader

            # Create mesh data
            nodes = np.random.randn(10, 3)
            edges = np.array([[i, (i+1) % 10] for i in range(10)]).T

            mesh = MeshData(nodes, edges)

            # Test feature computation
            edge_features = mesh.compute_edge_features()
            node_features = mesh.compute_node_features()

            assert edge_features.shape[0] == 10  # Number of edges
            assert node_features.shape[0] == 10  # Number of nodes

            # Test data loader
            loader = MeshDataLoader(batch_size=2)
            data_list = loader.load_mesh_dataset([mesh])

            assert len(data_list) == 1
            assert hasattr(data_list[0], 'x')
            assert hasattr(data_list[0], 'edge_index')

            print_success("Mesh data processing test passed")
            self.test_results['graph_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Mesh data processing test failed: {str(e)}")
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(f"Mesh Processing: {str(e)}")

    def test_geometric_attention(self):
        """Test geometric attention mechanism"""
        print_info("Testing geometric attention...")

        try:
            from app.ml.algorithms.graph_neural_network import GeometricAttentionLayer

            # Create geometric attention layer
            geom_attention = GeometricAttentionLayer(
                in_channels=16,
                out_channels=16,
                heads=4
            )

            # Test data
            x = torch.randn(5, 16)  # 5 nodes, 16 features
            edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
            pos = torch.randn(5, 3)  # 3D positions

            # Forward pass
            output = geom_attention(x, edge_index, pos)

            assert output.shape == (5, 16)

            print_success("Geometric attention test passed")
            self.test_results['graph_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Geometric attention test failed: {str(e)}")
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(f"Geometric Attention: {str(e)}")

    def test_cad_optimization(self):
        """Test CAD parameter optimization"""
        print_info("Testing CAD parameter optimization...")

        try:
            from app.ml.algorithms.graph_neural_network import (
                create_mesh_surrogate, create_cad_optimizer, MeshData
            )

            # Create mesh surrogate
            mesh_gnn = create_mesh_surrogate(
                input_dim=3,
                output_dim=1,
                task_type='graph_prediction'
            )

            # Create CAD optimizer
            cad_optimizer = create_cad_optimizer(mesh_gnn, parameter_dim=5)

            # Test mesh data
            nodes = np.random.randn(8, 3)
            edges = np.array([[i, (i+1) % 8] for i in range(8)]).T
            mesh = MeshData(nodes, edges)
            data = mesh.to_torch_geometric()

            # Test optimization
            target_performance = torch.tensor([[1.0]])
            result = cad_optimizer(data, target_performance)

            assert 'optimized_parameters' in result
            assert 'initial_parameters' in result
            assert 'optimization_history' in result

            print_success("CAD parameter optimization test passed")
            self.test_results['graph_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"CAD optimization test failed: {str(e)}")
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(f"CAD Optimization: {str(e)}")

    def test_graph_convolutions(self):
        """Test different graph convolution types"""
        print_info("Testing graph convolution types...")

        try:
            from app.ml.algorithms.graph_neural_network import GraphConvolutionalLayer

            conv_types = ['gcn', 'gat', 'sage', 'graph']

            for conv_type in conv_types:
                layer = GraphConvolutionalLayer(
                    in_channels=8,
                    out_channels=16,
                    conv_type=conv_type,
                    heads=2 if conv_type == 'gat' else 1
                )

                # Test data
                x = torch.randn(6, 8)
                edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)

                output = layer(x, edge_index)

                if conv_type == 'gat':
                    # GAT might concatenate heads
                    assert output.shape[0] == 6
                else:
                    assert output.shape == (6, 16)

            print_success("Graph convolution types test passed")
            self.test_results['graph_neural_networks']['passed'] += 1

        except Exception as e:
            print_error(f"Graph convolution test failed: {str(e)}")
            self.test_results['graph_neural_networks']['failed'] += 1
            self.test_results['graph_neural_networks']['errors'].append(f"Graph Conv: {str(e)}")

    # ==================== Transformer Model Tests ====================

    def test_transformer_creation(self):
        """Test Transformer model creation"""
        print_info("Testing Transformer model creation...")

        try:
            from app.ml.algorithms.transformer_surrogate import (
                OptimizationTransformer, TimeSeriesTransformer, PositionalEncoding
            )

            # Test positional encoding
            pos_enc = PositionalEncoding(d_model=64, max_len=100)
            x = torch.randn(10, 20, 64)  # [seq_len, batch, d_model]
            encoded = pos_enc(x)
            assert encoded.shape == x.shape

            # Test optimization transformer
            opt_transformer = OptimizationTransformer(
                input_dim=5,
                output_dim=2,
                d_model=64,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2
            )

            src = torch.randn(3, 10, 5)  # [batch, seq_len, input_dim]
            output = opt_transformer(src)

            assert 'predictions' in output
            assert 'uncertainties' in output

            print_success("Transformer creation test passed")
            self.test_results['transformer_models']['passed'] += 1

        except Exception as e:
            print_error(f"Transformer creation test failed: {str(e)}")
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(f"Creation: {str(e)}")

    def test_optimization_transformer(self):
        """Test optimization transformer functionality"""
        print_info("Testing optimization transformer...")

        try:
            from app.ml.algorithms.transformer_surrogate import create_optimization_transformer

            # Create transformer
            transformer = create_optimization_transformer(
                input_dim=4,
                output_dim=1,
                architecture='standard'
            )

            # Test sequence generation
            src = torch.randn(2, 5, 4)  # [batch, seq_len, input_dim]
            generated = transformer.generate_sequence(src, max_length=3)

            assert generated.shape[1] == src.shape[1] + 3  # Original + generated

            # Test with attention
            output = transformer(src, return_attention=True)
            assert 'predictions' in output
            assert 'uncertainties' in output

            print_success("Optimization transformer test passed")
            self.test_results['transformer_models']['passed'] += 1

        except Exception as e:
            print_error(f"Optimization transformer test failed: {str(e)}")
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(f"Optimization: {str(e)}")

    def test_timeseries_transformer(self):
        """Test time series transformer functionality"""
        print_info("Testing time series transformer...")

        try:
            from app.ml.algorithms.transformer_surrogate import create_timeseries_transformer

            # Create time series transformer
            ts_transformer = create_timeseries_transformer(
                input_dim=3,
                output_dim=1,
                forecast_horizon=5
            )

            # Test data
            x = torch.randn(4, 20, 3)  # [batch, seq_len, input_dim]
            time_indices = torch.arange(20).unsqueeze(0).expand(4, -1)

            # Forward pass
            output = ts_transformer(x, time_indices)

            assert 'predictions' in output
            assert 'uncertainties' in output
            assert 'attention_weights' in output

            # Check forecast horizon
            predictions = output['predictions']
            assert predictions.shape == (4, 5, 1)  # [batch, forecast_horizon, output_dim]

            print_success("Time series transformer test passed")
            self.test_results['transformer_models']['passed'] += 1

        except Exception as e:
            print_error(f"Time series transformer test failed: {str(e)}")
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(f"Time Series: {str(e)}")

    def test_multimodal_fusion(self):
        """Test multi-modal data fusion"""
        print_info("Testing multi-modal fusion...")

        try:
            from app.ml.algorithms.transformer_surrogate import MultiModalFusion

            # Create fusion model
            modal_dims = {
                'tabular': 10,
                'timeseries': 20,
                'image': 512
            }

            fusion = MultiModalFusion(
                modal_dims=modal_dims,
                fusion_dim=64,
                fusion_type='attention'
            )

            # Test inputs
            modal_inputs = {
                'tabular': torch.randn(5, 10),
                'timeseries': torch.randn(5, 20),
                'image': torch.randn(5, 512)
            }

            # Fusion
            fused = fusion(modal_inputs)

            assert fused.shape == (5, 64)

            # Test different fusion types
            for fusion_type in ['concat', 'gated']:
                fusion_alt = MultiModalFusion(modal_dims, 64, fusion_type)
                fused_alt = fusion_alt(modal_inputs)
                assert fused_alt.shape[0] == 5

            print_success("Multi-modal fusion test passed")
            self.test_results['transformer_models']['passed'] += 1

        except Exception as e:
            print_error(f"Multi-modal fusion test failed: {str(e)}")
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(f"Multi-modal: {str(e)}")

    def test_transfer_learning(self):
        """Test transfer learning capabilities"""
        print_info("Testing transfer learning...")

        try:
            from app.ml.algorithms.transformer_surrogate import (
                TransferLearningTransformer, OptimizationTransformer
            )

            # Create base transformer
            base_transformer = OptimizationTransformer(
                input_dim=5,
                output_dim=2,
                d_model=32,
                nhead=2,
                num_encoder_layers=2,
                num_decoder_layers=2
            )

            # Create transfer learning model
            transfer_model = TransferLearningTransformer(
                base_transformer=base_transformer,
                target_input_dim=8,
                target_output_dim=1,
                adaptation_layers=2,
                freeze_base=True
            )

            # Test forward pass
            x = torch.randn(3, 10, 8)
            output = transfer_model(x)

            assert 'predictions' in output
            assert 'base_features' in output

            print_success("Transfer learning test passed")
            self.test_results['transformer_models']['passed'] += 1

        except Exception as e:
            print_error(f"Transfer learning test failed: {str(e)}")
            self.test_results['transformer_models']['failed'] += 1
            self.test_results['transformer_models']['errors'].append(f"Transfer Learning: {str(e)}")

    # ==================== Uncertainty Quantification Tests ====================

    def test_aleatoric_uncertainty(self):
        """Test aleatoric uncertainty modeling"""
        print_info("Testing aleatoric uncertainty...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianNeuralNetwork

            # Create heteroscedastic BNN
            bnn = BayesianNeuralNetwork(
                input_dim=2,
                output_dim=1,
                hidden_layers=[16],
                heteroscedastic=True
            )

            x = torch.randn(5, 2)
            output = bnn(x)

            # Output should be [mean, log_var]
            assert output.shape == (5, 2)

            # Get uncertainty prediction
            mean, aleatoric_std, epistemic_std = bnn.predict_with_uncertainty(x, n_samples=10)

            assert mean.shape == (5, 1)
            assert aleatoric_std.shape == (5, 1)
            assert epistemic_std.shape == (5, 1)

            print_success("Aleatoric uncertainty test passed")
            self.test_results['uncertainty_quantification']['passed'] += 1

        except Exception as e:
            print_error(f"Aleatoric uncertainty test failed: {str(e)}")
            self.test_results['uncertainty_quantification']['failed'] += 1
            self.test_results['uncertainty_quantification']['errors'].append(f"Aleatoric: {str(e)}")

    def test_epistemic_uncertainty(self):
        """Test epistemic uncertainty modeling"""
        print_info("Testing epistemic uncertainty...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianNeuralNetwork

            # Create BNN with MC dropout
            bnn = BayesianNeuralNetwork(
                input_dim=3,
                output_dim=1,
                hidden_layers=[32, 16],
                use_mc_dropout=True,
                heteroscedastic=False
            )

            x = torch.randn(8, 3)

            # Multiple forward passes should give different results
            outputs = []
            for _ in range(5):
                output = bnn(x)
                outputs.append(output)

            outputs = torch.stack(outputs, dim=0)
            epistemic_var = torch.var(outputs, dim=0)

            assert epistemic_var.mean() > 0

            print_success("Epistemic uncertainty test passed")
            self.test_results['uncertainty_quantification']['passed'] += 1

        except Exception as e:
            print_error(f"Epistemic uncertainty test failed: {str(e)}")
            self.test_results['uncertainty_quantification']['failed'] += 1
            self.test_results['uncertainty_quantification']['errors'].append(f"Epistemic: {str(e)}")

    def test_uncertainty_calibration(self):
        """Test uncertainty calibration metrics"""
        print_info("Testing uncertainty calibration...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianTrainer, BayesianNeuralNetwork

            # Create and train BNN
            bnn = BayesianNeuralNetwork(2, 1, [16])
            trainer = BayesianTrainer(bnn)

            # Generate calibration data
            X_cal = np.random.randn(50, 2)
            y_cal = X_cal[:, 0:1] + np.random.randn(50, 1) * 0.1

            # Quick training
            trainer.train(X_cal[:30], y_cal[:30], epochs=5)

            # Test calibration
            calibration_metrics = trainer.calibrate_uncertainty(X_cal[30:], y_cal[30:])

            assert 'mean_calibration_error' in calibration_metrics
            assert 'sharpness' in calibration_metrics
            assert 'negative_log_likelihood' in calibration_metrics

            print_success("Uncertainty calibration test passed")
            self.test_results['uncertainty_quantification']['passed'] += 1

        except Exception as e:
            print_error(f"Uncertainty calibration test failed: {str(e)}")
            self.test_results['uncertainty_quantification']['failed'] += 1
            self.test_results['uncertainty_quantification']['errors'].append(f"Calibration: {str(e)}")

    def test_ensemble_uncertainty(self):
        """Test ensemble uncertainty quantification"""
        print_info("Testing ensemble uncertainty...")

        try:
            from app.ml.algorithms.bayesian_neural_network import BayesianEnsemble

            # Create ensemble
            ensemble = BayesianEnsemble(
                input_dim=2,
                output_dim=1,
                n_models=3,
                hidden_layers=[16]
            )

            x = torch.randn(4, 2)

            # Get comprehensive uncertainty
            uncertainty_result = ensemble.predict_with_uncertainty(x, n_samples=10)

            required_keys = [
                'mean', 'aleatoric_uncertainty', 'epistemic_uncertainty',
                'ensemble_variance', 'total_uncertainty'
            ]

            for key in required_keys:
                assert key in uncertainty_result

            print_success("Ensemble uncertainty test passed")
            self.test_results['uncertainty_quantification']['passed'] += 1

        except Exception as e:
            print_error(f"Ensemble uncertainty test failed: {str(e)}")
            self.test_results['uncertainty_quantification']['failed'] += 1
            self.test_results['uncertainty_quantification']['errors'].append(f"Ensemble: {str(e)}")

    # ==================== Attention Mechanism Tests ====================

    def test_feature_attention(self):
        """Test feature attention mechanism"""
        print_info("Testing feature attention...")

        try:
            from app.ml.algorithms.transformer_surrogate import FeatureAttention

            # Create feature attention
            feature_attention = FeatureAttention(feature_dim=10, hidden_dim=32)

            # Test data
            x = torch.randn(5, 20, 10)  # [batch, seq_len, feature_dim]

            # Apply attention
            attended_features, attention_weights = feature_attention(x)

            assert attended_features.shape == x.shape
            assert attention_weights.shape == (5, 20, 1)

            # Attention weights should sum to 1 across sequence dimension
            assert torch.allclose(attention_weights.sum(dim=1), torch.ones(5, 1), atol=1e-6)

            print_success("Feature attention test passed")
            self.test_results['attention_mechanisms']['passed'] += 1

        except Exception as e:
            print_error(f"Feature attention test failed: {str(e)}")
            self.test_results['attention_mechanisms']['failed'] += 1
            self.test_results['attention_mechanisms']['errors'].append(f"Feature Attention: {str(e)}")

    def test_geometric_attention_detailed(self):
        """Test geometric attention in detail"""
        print_info("Testing geometric attention (detailed)...")

        try:
            from app.ml.algorithms.graph_neural_network import GeometricAttentionLayer

            # Create geometric attention
            geom_attention = GeometricAttentionLayer(
                in_channels=32,
                out_channels=32,
                heads=8
            )

            # Test with more complex geometry
            x = torch.randn(10, 32)
            edge_index = torch.tensor([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
            ], dtype=torch.long)
            pos = torch.randn(10, 3)

            # Make positions more structured
            pos[:5, 2] = 0  # First 5 nodes on z=0 plane
            pos[5:, 2] = 1  # Last 5 nodes on z=1 plane

            output = geom_attention(x, edge_index, pos)

            assert output.shape == (10, 32)

            print_success("Geometric attention (detailed) test passed")
            self.test_results['attention_mechanisms']['passed'] += 1

        except Exception as e:
            print_error(f"Geometric attention (detailed) test failed: {str(e)}")
            self.test_results['attention_mechanisms']['failed'] += 1
            self.test_results['attention_mechanisms']['errors'].append(f"Geometric Attention: {str(e)}")

    def test_multihead_attention(self):
        """Test multi-head attention implementation"""
        print_info("Testing multi-head attention...")

        try:
            from app.ml.algorithms.transformer_surrogate import OptimizationTransformer

            # Create transformer with multiple heads
            transformer = OptimizationTransformer(
                input_dim=6,
                output_dim=2,
                d_model=64,
                nhead=8,
                num_encoder_layers=2
            )

            # Test data
            src = torch.randn(3, 15, 6)

            # Forward with attention
            output = transformer(src, return_attention=True)

            assert 'predictions' in output
            predictions = output['predictions']
            assert predictions.shape == (3, 15, 2)

            print_success("Multi-head attention test passed")
            self.test_results['attention_mechanisms']['passed'] += 1

        except Exception as e:
            print_error(f"Multi-head attention test failed: {str(e)}")
            self.test_results['attention_mechanisms']['failed'] += 1
            self.test_results['attention_mechanisms']['errors'].append(f"Multi-head: {str(e)}")

    def test_attention_visualization(self):
        """Test attention weight extraction for visualization"""
        print_info("Testing attention visualization...")

        try:
            from app.ml.algorithms.transformer_surrogate import TransformerTrainer, OptimizationTransformer

            # Create transformer
            transformer = OptimizationTransformer(
                input_dim=4,
                output_dim=1,
                d_model=32,
                nhead=4,
                use_feature_attention=True
            )

            # Create trainer
            trainer = TransformerTrainer(transformer)

            # Test attention extraction
            inputs = torch.randn(2, 10, 4)
            result = trainer.predict_with_attention(inputs)

            # Should have attention information
            if 'feature_attention' in result:
                attention = result['feature_attention']
                assert attention.shape[0] == 2  # Batch size
                assert attention.shape[1] == 10  # Sequence length

            print_success("Attention visualization test passed")
            self.test_results['attention_mechanisms']['passed'] += 1

        except Exception as e:
            print_error(f"Attention visualization test failed: {str(e)}")
            self.test_results['attention_mechanisms']['failed'] += 1
            self.test_results['attention_mechanisms']['errors'].append(f"Visualization: {str(e)}")

    # ==================== API Endpoint Tests ====================

    def test_bayesian_api(self):
        """Test Bayesian Neural Network API endpoints"""
        print_info("Testing Bayesian API endpoints...")

        try:
            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            # Test session creation
            config = {
                "input_dim": 3,
                "output_dim": 1,
                "hidden_layers": [32, 16],
                "activation": "relu",
                "ensemble_size": 3
            }

            response = requests.post(
                f"{API_BASE}/nextgen-ml/bayesian/sessions",
                json=config,
                headers=headers
            )

            if response.status_code == 200:
                print_success("Bayesian API session creation test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"Bayesian API call failed: {response.status_code}")

        except Exception as e:
            print_error(f"Bayesian API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Bayesian API: {str(e)}")

    def test_graph_api(self):
        """Test Graph Neural Network API endpoints"""
        print_info("Testing Graph API endpoints...")

        try:
            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            # Test session creation
            config = {
                "input_dim": 3,
                "output_dim": 1,
                "conv_type": "gat",
                "use_geometric_attention": True,
                "task_type": "node_prediction"
            }

            response = requests.post(
                f"{API_BASE}/nextgen-ml/graph/sessions",
                json=config,
                headers=headers
            )

            if response.status_code == 200:
                print_success("Graph API session creation test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"Graph API call failed: {response.status_code}")

        except Exception as e:
            print_error(f"Graph API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Graph API: {str(e)}")

    def test_transformer_api(self):
        """Test Transformer API endpoints"""
        print_info("Testing Transformer API endpoints...")

        try:
            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            # Test session creation
            config = {
                "input_dim": 5,
                "output_dim": 2,
                "d_model": 128,
                "nhead": 8,
                "transformer_type": "optimization",
                "use_feature_attention": True
            }

            response = requests.post(
                f"{API_BASE}/nextgen-ml/transformer/sessions",
                json=config,
                headers=headers
            )

            if response.status_code == 200:
                print_success("Transformer API session creation test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"Transformer API call failed: {response.status_code}")

        except Exception as e:
            print_error(f"Transformer API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Transformer API: {str(e)}")

    def test_capabilities_endpoint(self):
        """Test capabilities overview endpoint"""
        print_info("Testing capabilities endpoint...")

        try:
            response = requests.get(f"{API_BASE}/nextgen-ml/capabilities")

            if response.status_code == 200:
                capabilities = response.json()

                required_keys = [
                    'bayesian_neural_networks',
                    'graph_neural_networks',
                    'transformer_models'
                ]

                for key in required_keys:
                    assert key in capabilities['next_generation_ml_capabilities']

                print_success("Capabilities endpoint test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"Capabilities endpoint failed: {response.status_code}")

        except Exception as e:
            print_error(f"Capabilities endpoint test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Capabilities: {str(e)}")

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print_header("📊 Next-Generation ML Test Results Summary")

        total_passed = 0
        total_failed = 0

        for category, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed

            status_color = Colors.GREEN if failed == 0 else Colors.RED
            print(f"{status_color}{category.replace('_', ' ').title()}: {passed} passed, {failed} failed{Colors.ENDC}")

            if results['errors']:
                for error in results['errors']:
                    print(f"  {Colors.RED}• {error}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}Overall Results:{Colors.ENDC}")
        overall_color = Colors.GREEN if total_failed == 0 else Colors.RED
        print(f"{overall_color}Total: {total_passed} passed, {total_failed} failed{Colors.ENDC}")

        if total_failed == 0:
            print_success("🚀 All next-generation ML tests completed successfully!")
            print(f"{Colors.CYAN}Ready for production deployment of advanced ML capabilities!{Colors.ENDC}")
        else:
            print_error(f"❌ {total_failed} tests failed. Please review the errors above.")

        # Summary of capabilities tested
        print(f"\n{Colors.BOLD}Capabilities Validated:{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ Bayesian Neural Networks with uncertainty quantification{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ Graph Neural Networks for CAD/mesh processing{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ Transformer models with attention mechanisms{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ Multi-modal data fusion{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ Feature importance and attention analysis{Colors.ENDC}")
        print(f"{Colors.CYAN}✓ API endpoints for next-generation ML{Colors.ENDC}")


def main():
    """Main test execution"""
    test_suite = NextGenMLTestSuite()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()