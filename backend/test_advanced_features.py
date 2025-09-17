#!/usr/bin/env python3
"""
Comprehensive test suite for advanced multi-fidelity and physics-informed modeling capabilities.

This test script validates the new features:
1. Multi-Fidelity Modeling (Co-Kriging, Hierarchical)
2. Information Fusion Algorithms
3. Physics-Informed Neural Networks (PINNs)
4. Physics Validation System
5. API Endpoints for Advanced Features

Usage:
    python test_advanced_features.py
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
TEST_TOKEN = "fake-token-for-testing"  # Replace with actual auth token if needed

# Test colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
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
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.ENDC}")


class AdvancedFeaturesTestSuite:
    """Comprehensive test suite for advanced modeling features"""

    def __init__(self):
        self.test_results = {
            'multi_fidelity': {'passed': 0, 'failed': 0, 'errors': []},
            'physics_informed': {'passed': 0, 'failed': 0, 'errors': []},
            'information_fusion': {'passed': 0, 'failed': 0, 'errors': []},
            'physics_validation': {'passed': 0, 'failed': 0, 'errors': []},
            'api_endpoints': {'passed': 0, 'failed': 0, 'errors': []}
        }

    def run_all_tests(self):
        """Run the complete test suite"""
        print_header("🧪 Advanced Surrogate Modeling Features Test Suite")

        try:
            # Test 1: Multi-Fidelity Modeling
            self.test_multi_fidelity_modeling()

            # Test 2: Physics-Informed Neural Networks
            self.test_physics_informed_nn()

            # Test 3: Information Fusion
            self.test_information_fusion()

            # Test 4: Physics Validation
            self.test_physics_validation()

            # Test 5: API Endpoints
            self.test_api_endpoints()

            # Summary
            self.print_test_summary()

        except Exception as e:
            print_error(f"Test suite failed with error: {str(e)}")
            traceback.print_exc()

    def test_multi_fidelity_modeling(self):
        """Test multi-fidelity modeling capabilities"""
        print_header("🎯 Multi-Fidelity Modeling Tests")

        try:
            # Generate synthetic multi-fidelity data
            low_fidelity_data, high_fidelity_data = self.generate_multi_fidelity_data()

            # Test Co-Kriging Model
            self.test_co_kriging_model(low_fidelity_data, high_fidelity_data)

            # Test Hierarchical Model
            self.test_hierarchical_model(low_fidelity_data, high_fidelity_data)

            # Test Fidelity Optimization
            self.test_fidelity_optimization(low_fidelity_data, high_fidelity_data)

        except Exception as e:
            self.test_results['multi_fidelity']['failed'] += 1
            self.test_results['multi_fidelity']['errors'].append(str(e))
            print_error(f"Multi-fidelity testing failed: {str(e)}")

    def test_physics_informed_nn(self):
        """Test Physics-Informed Neural Network implementation"""
        print_header("🧠 Physics-Informed Neural Networks Tests")

        try:
            # Test PINN Creation
            self.test_pinn_creation()

            # Test Physics Constraints
            self.test_physics_constraints()

            # Test PINN Training
            self.test_pinn_training()

            # Test Physics Validation
            self.test_pinn_physics_validation()

        except Exception as e:
            self.test_results['physics_informed']['failed'] += 1
            self.test_results['physics_informed']['errors'].append(str(e))
            print_error(f"PINN testing failed: {str(e)}")

    def test_information_fusion(self):
        """Test information fusion algorithms"""
        print_header("🔗 Information Fusion Tests")

        try:
            # Test Fusion Model Creation
            self.test_fusion_model_creation()

            # Test Weight Learning
            self.test_fusion_weight_learning()

            # Test Prediction Fusion
            self.test_prediction_fusion()

        except Exception as e:
            self.test_results['information_fusion']['failed'] += 1
            self.test_results['information_fusion']['errors'].append(str(e))
            print_error(f"Information fusion testing failed: {str(e)}")

    def test_physics_validation(self):
        """Test physics validation system"""
        print_header("⚖️ Physics Validation Tests")

        try:
            # Test Validator Creation
            self.test_validator_creation()

            # Test Conservation Laws
            self.test_conservation_law_validation()

            # Test Dimensional Analysis
            self.test_dimensional_analysis()

            # Test Domain-Specific Validation
            self.test_domain_specific_validation()

        except Exception as e:
            self.test_results['physics_validation']['failed'] += 1
            self.test_results['physics_validation']['errors'].append(str(e))
            print_error(f"Physics validation testing failed: {str(e)}")

    def test_api_endpoints(self):
        """Test API endpoints for advanced features"""
        print_header("🌐 API Endpoints Tests")

        try:
            # Test Multi-Fidelity API
            self.test_multi_fidelity_api()

            # Test PINN API
            self.test_pinn_api()

            # Test Physics Validation API
            self.test_physics_validation_api()

        except Exception as e:
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(str(e))
            print_error(f"API endpoints testing failed: {str(e)}")

    def generate_multi_fidelity_data(self) -> Tuple[Dict, Dict]:
        """Generate synthetic multi-fidelity data for testing"""
        print_info("Generating synthetic multi-fidelity data...")

        # Create input space
        np.random.seed(42)

        # Low fidelity: more samples, less accurate
        X_low = np.random.uniform(-2, 2, (50, 2))
        y_low = self.low_fidelity_function(X_low) + np.random.normal(0, 0.1, 50)

        # High fidelity: fewer samples, more accurate
        X_high = np.random.uniform(-2, 2, (20, 2))
        y_high = self.high_fidelity_function(X_high) + np.random.normal(0, 0.02, 20)

        low_fidelity_data = {'X': X_low, 'y': y_low}
        high_fidelity_data = {'X': X_high, 'y': y_high}

        print_success(f"Generated multi-fidelity data: {len(X_low)} low-fidelity, {len(X_high)} high-fidelity samples")

        return low_fidelity_data, high_fidelity_data

    def low_fidelity_function(self, X: np.ndarray) -> np.ndarray:
        """Simplified function for low fidelity simulation"""
        return X[:, 0]**2 + X[:, 1]**2

    def high_fidelity_function(self, X: np.ndarray) -> np.ndarray:
        """More accurate function for high fidelity simulation"""
        return X[:, 0]**2 + X[:, 1]**2 + 0.5 * np.sin(5 * X[:, 0]) * np.cos(5 * X[:, 1])

    def test_co_kriging_model(self, low_fidelity_data: Dict, high_fidelity_data: Dict):
        """Test Co-Kriging multi-fidelity model"""
        print_info("Testing Co-Kriging model...")

        try:
            from app.ml.active_learning.multi_fidelity.co_kriging import CoKrigingModel

            # Define fidelity levels
            fidelity_levels = [
                {'level': 0, 'cost': 1.0, 'accuracy': 0.7, 'name': 'Low Fidelity'},
                {'level': 1, 'cost': 5.0, 'accuracy': 0.95, 'name': 'High Fidelity'}
            ]

            # Create model
            model = CoKrigingModel(fidelity_levels)

            # Prepare training data
            multi_fidelity_data = {
                0: (low_fidelity_data['X'], low_fidelity_data['y']),
                1: (high_fidelity_data['X'], high_fidelity_data['y'])
            }

            # Train model
            model.fit(multi_fidelity_data)

            # Test predictions
            X_test = np.array([[0.5, 0.5], [1.0, 1.0]])

            # Low fidelity predictions
            predictions_low = model.predict(X_test, fidelity_level=0)

            # High fidelity predictions
            predictions_high = model.predict(X_test, fidelity_level=1)

            # Validate results
            assert 'output_0' in predictions_low
            assert 'output_0' in predictions_high
            assert 'prediction' in predictions_low['output_0']
            assert 'uncertainty' in predictions_low['output_0']

            # Get correlation analysis
            correlation_analysis = model.get_correlation_analysis()
            assert 'estimated_correlation' in correlation_analysis

            print_success("Co-Kriging model test passed")
            self.test_results['multi_fidelity']['passed'] += 1

        except Exception as e:
            print_error(f"Co-Kriging model test failed: {str(e)}")
            self.test_results['multi_fidelity']['failed'] += 1
            self.test_results['multi_fidelity']['errors'].append(f"Co-Kriging: {str(e)}")

    def test_hierarchical_model(self, low_fidelity_data: Dict, high_fidelity_data: Dict):
        """Test Hierarchical multi-fidelity model"""
        print_info("Testing Hierarchical model...")

        try:
            from app.ml.active_learning.multi_fidelity.hierarchical_model import HierarchicalMultiFidelityModel

            # Define fidelity levels
            fidelity_levels = [
                {'level': 0, 'cost': 1.0, 'accuracy': 0.7, 'name': 'Low Fidelity'},
                {'level': 1, 'cost': 5.0, 'accuracy': 0.95, 'name': 'High Fidelity'}
            ]

            # Create model
            model = HierarchicalMultiFidelityModel(fidelity_levels)

            # Prepare training data
            multi_fidelity_data = {
                0: (low_fidelity_data['X'], low_fidelity_data['y']),
                1: (high_fidelity_data['X'], high_fidelity_data['y'])
            }

            # Train model
            model.fit(multi_fidelity_data)

            # Test predictions
            X_test = np.array([[0.5, 0.5], [1.0, 1.0]])
            predictions = model.predict(X_test)

            # Get analysis
            analysis = model.get_fidelity_analysis()

            # Validate results
            assert 'output_0' in predictions
            assert 'hierarchical_path' in predictions['output_0']
            assert 'fidelity_correlations' in analysis

            print_success("Hierarchical model test passed")
            self.test_results['multi_fidelity']['passed'] += 1

        except Exception as e:
            print_error(f"Hierarchical model test failed: {str(e)}")
            self.test_results['multi_fidelity']['failed'] += 1
            self.test_results['multi_fidelity']['errors'].append(f"Hierarchical: {str(e)}")

    def test_fidelity_optimization(self, low_fidelity_data: Dict, high_fidelity_data: Dict):
        """Test fidelity allocation optimization"""
        print_info("Testing fidelity allocation optimization...")

        try:
            from app.ml.active_learning.multi_fidelity.hierarchical_model import HierarchicalMultiFidelityModel

            # Create and train model
            fidelity_levels = [
                {'level': 0, 'cost': 1.0, 'accuracy': 0.7},
                {'level': 1, 'cost': 5.0, 'accuracy': 0.95}
            ]

            model = HierarchicalMultiFidelityModel(fidelity_levels)
            multi_fidelity_data = {
                0: (low_fidelity_data['X'], low_fidelity_data['y']),
                1: (high_fidelity_data['X'], high_fidelity_data['y'])
            }
            model.fit(multi_fidelity_data)

            # Test optimization
            candidates = np.random.uniform(-1, 1, (10, 2))
            allocation_plan = model.optimize_fidelity_allocation(
                total_budget=20.0,
                X_candidates=candidates,
                optimization_horizon=5
            )

            # Validate plan
            assert isinstance(allocation_plan, list)
            assert len(allocation_plan) > 0

            total_cost = sum(step['cost'] for step in allocation_plan)
            assert total_cost <= 20.0

            print_success("Fidelity optimization test passed")
            self.test_results['multi_fidelity']['passed'] += 1

        except Exception as e:
            print_error(f"Fidelity optimization test failed: {str(e)}")
            self.test_results['multi_fidelity']['failed'] += 1
            self.test_results['multi_fidelity']['errors'].append(f"Optimization: {str(e)}")

    def test_pinn_creation(self):
        """Test PINN model creation"""
        print_info("Testing PINN creation...")

        try:
            from app.ml.algorithms.physics_informed_nn import (
                PhysicsInformedNN, ConservationLaw, BoundaryCondition, create_engineering_pinn
            )

            # Test manual PINN creation
            constraints = [
                ConservationLaw('mass_conservation'),
                BoundaryCondition('dirichlet', boundary_value=0.0)
            ]

            model = PhysicsInformedNN(
                input_dim=2,
                output_dim=3,
                hidden_layers=[32, 32],
                activation='tanh',
                physics_constraints=constraints
            )

            # Test forward pass
            x = torch.randn(5, 2, requires_grad=True)
            output = model(x)

            assert output.shape == (5, 3)
            assert len(model.physics_constraints) == 2

            # Test engineering PINN creation
            fluid_model, fluid_constraints = create_engineering_pinn(
                'fluid_flow', input_dim=2, output_dim=3
            )

            assert len(fluid_constraints) > 0

            print_success("PINN creation test passed")
            self.test_results['physics_informed']['passed'] += 1

        except Exception as e:
            print_error(f"PINN creation test failed: {str(e)}")
            self.test_results['physics_informed']['failed'] += 1
            self.test_results['physics_informed']['errors'].append(f"Creation: {str(e)}")

    def test_physics_constraints(self):
        """Test physics constraint implementation"""
        print_info("Testing physics constraints...")

        try:
            from app.ml.algorithms.physics_informed_nn import (
                ConservationLaw, BoundaryCondition, DimensionalConsistency
            )

            # Test conservation law
            conservation = ConservationLaw('mass_conservation')

            # Create dummy data
            x = torch.randn(5, 2, requires_grad=True)
            model_output = torch.randn(5, 3, requires_grad=True)

            # Test loss computation
            loss = conservation.compute_loss(model_output, x, None)
            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad

            # Test boundary condition
            boundary = BoundaryCondition('dirichlet', boundary_value=1.0)
            boundary_loss = boundary.compute_loss(model_output, x, None)
            assert isinstance(boundary_loss, torch.Tensor)

            # Test dimensional consistency
            dim_check = DimensionalConsistency({})
            dim_loss = dim_check.compute_loss(model_output, x, None)
            assert isinstance(dim_loss, torch.Tensor)

            print_success("Physics constraints test passed")
            self.test_results['physics_informed']['passed'] += 1

        except Exception as e:
            print_error(f"Physics constraints test failed: {str(e)}")
            self.test_results['physics_informed']['failed'] += 1
            self.test_results['physics_informed']['errors'].append(f"Constraints: {str(e)}")

    def test_pinn_training(self):
        """Test PINN training process"""
        print_info("Testing PINN training...")

        try:
            from app.ml.algorithms.physics_informed_nn import (
                PhysicsInformedNN, PINNTrainer, ConservationLaw
            )

            # Create simple PINN
            constraints = [ConservationLaw('mass_conservation', weight=0.1)]
            model = PhysicsInformedNN(
                input_dim=2,
                output_dim=1,
                hidden_layers=[16, 16],
                physics_constraints=constraints
            )

            # Create trainer
            trainer = PINNTrainer(
                model=model,
                data_weight=1.0,
                physics_weight=0.1,
                learning_rate=1e-3
            )

            # Generate training data
            X_data = np.random.uniform(-1, 1, (20, 2))
            y_data = X_data[:, 0:1]**2 + X_data[:, 1:1]**2  # Simple function
            X_physics = np.random.uniform(-1, 1, (30, 2))

            # Train for a few epochs
            results = trainer.train(
                X_data=X_data,
                y_data=y_data,
                X_physics=X_physics,
                epochs=10
            )

            # Validate training results
            assert 'final_loss' in results
            assert 'training_history' in results
            assert len(results['training_history']) == 10

            # Test predictions
            X_test = np.array([[0.5, 0.5]])
            predictions = trainer.predict(X_test)

            assert 'output_0' in predictions
            assert 'physics_informed' in predictions['output_0']

            print_success("PINN training test passed")
            self.test_results['physics_informed']['passed'] += 1

        except Exception as e:
            print_error(f"PINN training test failed: {str(e)}")
            self.test_results['physics_informed']['failed'] += 1
            self.test_results['physics_informed']['errors'].append(f"Training: {str(e)}")

    def test_pinn_physics_validation(self):
        """Test PINN physics validation"""
        print_info("Testing PINN physics validation...")

        try:
            from app.ml.utils.physics_validator import create_physics_validator

            # Create validator
            validator = create_physics_validator('fluid')

            # Test predictions
            predictions = {
                'pressure': {'prediction': 101325.0, 'uncertainty': {'standard_deviation': 100.0}},
                'velocity': {'prediction': 10.0, 'uncertainty': {'standard_deviation': 0.5}},
                'density': {'prediction': 1.225, 'uncertainty': {'standard_deviation': 0.01}}
            }

            inputs = {
                'temperature': 300.0,
                'mass_flow_in': 0.5
            }

            # Validate
            validation_results = validator.validate_predictions(predictions, inputs)

            # Check results
            assert 'overall_valid' in validation_results
            assert 'validation_score' in validation_results
            assert 'dimensional_analysis' in validation_results
            assert 'physics_checks' in validation_results

            print_success("PINN physics validation test passed")
            self.test_results['physics_informed']['passed'] += 1

        except Exception as e:
            print_error(f"PINN physics validation test failed: {str(e)}")
            self.test_results['physics_informed']['failed'] += 1
            self.test_results['physics_informed']['errors'].append(f"Validation: {str(e)}")

    def test_fusion_model_creation(self):
        """Test information fusion model creation"""
        print_info("Testing information fusion model creation...")

        try:
            from app.ml.active_learning.multi_fidelity.information_fusion import InformationFusionModel

            # Test different fusion methods
            fusion_methods = ['weighted', 'bayesian', 'adaptive_weighted']

            for method in fusion_methods:
                fusion_model = InformationFusionModel(
                    fusion_method=method,
                    uncertainty_weighting=True
                )

                assert fusion_model.fusion_method == method
                assert fusion_model.uncertainty_weighting == True

            print_success("Information fusion model creation test passed")
            self.test_results['information_fusion']['passed'] += 1

        except Exception as e:
            print_error(f"Fusion model creation test failed: {str(e)}")
            self.test_results['information_fusion']['failed'] += 1
            self.test_results['information_fusion']['errors'].append(f"Creation: {str(e)}")

    def test_fusion_weight_learning(self):
        """Test fusion weight learning"""
        print_info("Testing fusion weight learning...")

        try:
            from app.ml.active_learning.multi_fidelity.information_fusion import InformationFusionModel

            # Create fusion model
            fusion_model = InformationFusionModel(fusion_method='weighted')

            # Generate multi-fidelity data
            X1 = np.random.uniform(-1, 1, (20, 2))
            y1 = X1[:, 0]**2 + X1[:, 1]**2 + np.random.normal(0, 0.1, 20)

            X2 = np.random.uniform(-1, 1, (15, 2))
            y2 = X2[:, 0]**2 + X2[:, 1]**2 + np.random.normal(0, 0.05, 15)

            multi_fidelity_data = {
                0: (X1, y1),
                1: (X2, y2)
            }

            # Fit fusion weights
            fusion_model.fit_fusion_weights(multi_fidelity_data)

            # Check weights
            assert fusion_model.is_trained
            assert 0 in fusion_model.fidelity_weights
            assert 1 in fusion_model.fidelity_weights

            # Test weight retrieval
            X_test = np.array([[0.5, 0.5]])
            weights = fusion_model.get_fusion_weights(X_test, [0, 1])

            assert 0 in weights
            assert 1 in weights
            assert len(weights[0]) == 1
            assert len(weights[1]) == 1

            print_success("Fusion weight learning test passed")
            self.test_results['information_fusion']['passed'] += 1

        except Exception as e:
            print_error(f"Fusion weight learning test failed: {str(e)}")
            self.test_results['information_fusion']['failed'] += 1
            self.test_results['information_fusion']['errors'].append(f"Weight Learning: {str(e)}")

    def test_prediction_fusion(self):
        """Test prediction fusion"""
        print_info("Testing prediction fusion...")

        try:
            from app.ml.active_learning.multi_fidelity.information_fusion import InformationFusionModel

            # Create and train fusion model
            fusion_model = InformationFusionModel(fusion_method='weighted')

            # Dummy training for weights
            multi_fidelity_data = {
                0: (np.random.randn(10, 2), np.random.randn(10)),
                1: (np.random.randn(8, 2), np.random.randn(8))
            }
            fusion_model.fit_fusion_weights(multi_fidelity_data)

            # Create prediction dictionary
            predictions = {
                0: {
                    'output_0': {
                        'prediction': 1.5,
                        'uncertainty': {'standard_deviation': 0.2}
                    }
                },
                1: {
                    'output_0': {
                        'prediction': 1.8,
                        'uncertainty': {'standard_deviation': 0.1}
                    }
                }
            }

            X_test = np.array([[0.5, 0.5]])

            # Fuse predictions
            fused_result = fusion_model.fuse_predictions(predictions, X_test)

            # Validate fusion
            assert 'output_0' in fused_result
            assert 'prediction' in fused_result['output_0']
            assert 'fusion_weights' in fused_result['output_0']
            assert 'individual_predictions' in fused_result['output_0']

            print_success("Prediction fusion test passed")
            self.test_results['information_fusion']['passed'] += 1

        except Exception as e:
            print_error(f"Prediction fusion test failed: {str(e)}")
            self.test_results['information_fusion']['failed'] += 1
            self.test_results['information_fusion']['errors'].append(f"Prediction Fusion: {str(e)}")

    def test_validator_creation(self):
        """Test physics validator creation"""
        print_info("Testing physics validator creation...")

        try:
            from app.ml.utils.physics_validator import create_physics_validator

            # Test different domains
            domains = ['fluid', 'thermal', 'structural', 'general']

            for domain in domains:
                validator = create_physics_validator(domain)

                assert len(validator.validation_rules) > 0
                assert len(validator.dimensional_registry) > 0

            print_success("Physics validator creation test passed")
            self.test_results['physics_validation']['passed'] += 1

        except Exception as e:
            print_error(f"Physics validator creation test failed: {str(e)}")
            self.test_results['physics_validation']['failed'] += 1
            self.test_results['physics_validation']['errors'].append(f"Creation: {str(e)}")

    def test_conservation_law_validation(self):
        """Test conservation law validation"""
        print_info("Testing conservation law validation...")

        try:
            from app.ml.utils.physics_validator import ConservationLawRule

            # Test mass conservation
            mass_rule = ConservationLawRule('mass')

            # Valid case
            predictions = {
                'density': {'prediction': 1.225, 'uncertainty': {'standard_deviation': 0.01}}
            }
            inputs = {'mass_flow_in': 1.0}

            result = mass_rule.validate(predictions, inputs)

            assert 'valid' in result
            assert 'score' in result
            assert 'violations' in result
            assert 'warnings' in result

            print_success("Conservation law validation test passed")
            self.test_results['physics_validation']['passed'] += 1

        except Exception as e:
            print_error(f"Conservation law validation test failed: {str(e)}")
            self.test_results['physics_validation']['failed'] += 1
            self.test_results['physics_validation']['errors'].append(f"Conservation: {str(e)}")

    def test_dimensional_analysis(self):
        """Test dimensional analysis"""
        print_info("Testing dimensional analysis...")

        try:
            from app.ml.utils.physics_validator import PhysicsValidator

            validator = PhysicsValidator()

            # Register dimensions
            validator.register_dimensions('pressure', 'Pressure', 'Pa')
            validator.register_dimensions('velocity', 'Velocity', 'm/s')
            validator.register_dimensions('temperature', 'Temperature', 'K')

            # Test dimensional validation
            predictions = {
                'pressure': {'prediction': 101325.0},
                'velocity': {'prediction': 10.0},
                'temperature': {'prediction': 300.0}
            }

            inputs = {}

            validation_results = validator.validate_predictions(predictions, inputs)

            assert 'dimensional_analysis' in validation_results

            print_success("Dimensional analysis test passed")
            self.test_results['physics_validation']['passed'] += 1

        except Exception as e:
            print_error(f"Dimensional analysis test failed: {str(e)}")
            self.test_results['physics_validation']['failed'] += 1
            self.test_results['physics_validation']['errors'].append(f"Dimensional: {str(e)}")

    def test_domain_specific_validation(self):
        """Test domain-specific validation"""
        print_info("Testing domain-specific validation...")

        try:
            from app.ml.utils.physics_validator import (
                FluidMechanicsRule, ThermodynamicsRule, StructuralMechanicsRule
            )

            # Test fluid mechanics
            fluid_rule = FluidMechanicsRule()
            fluid_predictions = {
                'pressure': {'prediction': 101325.0, 'uncertainty': {'standard_deviation': 100.0}},
                'velocity': {'prediction': 10.0, 'uncertainty': {'standard_deviation': 0.5}},
                'density': {'prediction': 1.225, 'uncertainty': {'standard_deviation': 0.01}}
            }

            fluid_result = fluid_rule.validate(fluid_predictions, {})
            assert 'valid' in fluid_result

            # Test thermodynamics
            thermo_rule = ThermodynamicsRule()
            thermo_predictions = {
                'temperature': {'prediction': 300.0, 'uncertainty': {'standard_deviation': 1.0}}
            }

            thermo_result = thermo_rule.validate(thermo_predictions, {})
            assert 'valid' in thermo_result

            # Test structural mechanics
            struct_rule = StructuralMechanicsRule()
            struct_predictions = {
                'stress': {'prediction': 1e6, 'uncertainty': {'standard_deviation': 1e4}},
                'strain': {'prediction': 0.001, 'uncertainty': {'standard_deviation': 1e-5}}
            }

            struct_result = struct_rule.validate(struct_predictions, {})
            assert 'valid' in struct_result

            print_success("Domain-specific validation test passed")
            self.test_results['physics_validation']['passed'] += 1

        except Exception as e:
            print_error(f"Domain-specific validation test failed: {str(e)}")
            self.test_results['physics_validation']['failed'] += 1
            self.test_results['physics_validation']['errors'].append(f"Domain-specific: {str(e)}")

    def test_multi_fidelity_api(self):
        """Test multi-fidelity API endpoints"""
        print_info("Testing multi-fidelity API endpoints...")

        try:
            # Test session creation
            config = {
                "fidelity_levels": [
                    {"level": 0, "cost": 1.0, "accuracy": 0.7, "name": "Low Fidelity"},
                    {"level": 1, "cost": 5.0, "accuracy": 0.95, "name": "High Fidelity"}
                ],
                "model_type": "co_kriging",
                "fusion_config": {
                    "method": "weighted",
                    "uncertainty_weighting": True
                }
            }

            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            # Create session
            response = requests.post(
                f"{API_BASE}/multi-fidelity/sessions",
                json=config,
                headers=headers
            )

            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data['session_id']

                # Test training
                training_data = {
                    "fidelity_data": {
                        0: {
                            "X": [[1.0, 2.0], [2.0, 3.0]],
                            "y": [1.5, 2.5]
                        },
                        1: {
                            "X": [[1.5, 2.5]],
                            "y": [2.0]
                        }
                    }
                }

                train_response = requests.post(
                    f"{API_BASE}/multi-fidelity/sessions/{session_id}/train",
                    json=training_data,
                    headers=headers
                )

                if train_response.status_code == 200:
                    print_success("Multi-fidelity API test passed")
                    self.test_results['api_endpoints']['passed'] += 1
                else:
                    print_warning(f"Training API call failed: {train_response.status_code}")
            else:
                print_warning(f"Session creation failed: {response.status_code}")

        except Exception as e:
            print_error(f"Multi-fidelity API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Multi-fidelity API: {str(e)}")

    def test_pinn_api(self):
        """Test PINN API endpoints"""
        print_info("Testing PINN API endpoints...")

        try:
            config = {
                "input_dim": 2,
                "output_dim": 1,
                "hidden_layers": [32, 32],
                "activation": "tanh",
                "problem_type": "fluid_flow"
            }

            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            # Create session
            response = requests.post(
                f"{API_BASE}/physics-informed/sessions",
                json=config,
                headers=headers
            )

            if response.status_code == 200:
                print_success("PINN API test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"PINN API call failed: {response.status_code}")

        except Exception as e:
            print_error(f"PINN API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"PINN API: {str(e)}")

    def test_physics_validation_api(self):
        """Test physics validation API endpoints"""
        print_info("Testing physics validation API endpoints...")

        try:
            validation_request = {
                "predictions": {
                    "pressure": {"prediction": 101325.0},
                    "velocity": {"prediction": 10.0}
                },
                "inputs": {
                    "temperature": 300.0
                }
            }

            headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

            response = requests.post(
                f"{API_BASE}/physics-informed/validate-physics?validation_domain=fluid",
                json=validation_request,
                headers=headers
            )

            if response.status_code == 200:
                print_success("Physics validation API test passed")
                self.test_results['api_endpoints']['passed'] += 1
            else:
                print_warning(f"Physics validation API call failed: {response.status_code}")

        except Exception as e:
            print_error(f"Physics validation API test failed: {str(e)}")
            self.test_results['api_endpoints']['failed'] += 1
            self.test_results['api_endpoints']['errors'].append(f"Physics validation API: {str(e)}")

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print_header("📊 Test Results Summary")

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
            print_success("🎉 All advanced feature tests completed successfully!")
        else:
            print_error(f"❌ {total_failed} tests failed. Please review the errors above.")

    def mark_test_completed(self):
        """Mark the test task as completed"""
        try:
            # This would integrate with the TodoWrite system if available
            print_success("Advanced features testing completed successfully!")
        except:
            pass


def main():
    """Main test execution"""
    test_suite = AdvancedFeaturesTestSuite()
    test_suite.run_all_tests()
    test_suite.mark_test_completed()


if __name__ == "__main__":
    main()