# 🧪 Test Data for Surrogate Model Platform

This directory contains sample datasets and test scripts for the Surrogate Model Platform, demonstrating various engineering applications and machine learning model types.

## 📁 Contents

### Sample Datasets

#### 1. **Airfoil Performance** (`airfoil_performance.csv`)
- **Application**: Aerodynamic analysis
- **Features**: angle_of_attack, chord_length, Reynolds_number, thickness_ratio
- **Targets**: lift_coefficient, drag_coefficient
- **Use Case**: Bayesian Neural Networks with uncertainty quantification
- **Size**: 30 samples

#### 2. **Structural Analysis** (`structural_analysis.csv`)
- **Application**: Mechanical engineering
- **Features**: load_force, material_young_modulus, cross_section_area, length
- **Targets**: displacement, stress
- **Use Case**: Classical regression, Bayesian uncertainty
- **Size**: 25 samples

#### 3. **Heat Transfer** (`heat_transfer.csv`)
- **Application**: Thermal engineering
- **Features**: temperature_inlet, flow_rate, thermal_conductivity, surface_area
- **Targets**: temperature_outlet, heat_transfer_rate
- **Use Case**: Multi-output regression with thermal physics
- **Size**: 25 samples

### Extended Datasets (Generated)

When you run `generate_synthetic_data.py`, you'll get:

- **`airfoil_performance_extended.csv`**: 200 samples with noise and physics-based relationships
- **`structural_analysis_extended.csv`**: 150 samples with realistic material properties
- **`heat_transfer_extended.csv`**: 180 samples with thermal dynamics
- **`mesh_nodes.csv`**: 100 nodes for Graph Neural Network testing
- **`optimization_sequence.csv`**: 100 iterations for Transformer model testing

## 🚀 Quick Start

### 1. Run API Tests
```bash
cd /Users/durai/Documents/GitHub/SurrogateModel/test_data
./test_api_endpoints.sh
```

### 2. Generate Extended Datasets
```bash
python3 generate_synthetic_data.py
```

### 3. Test Different Model Types

#### Bayesian Neural Networks
```bash
curl -X POST -H "Authorization: Bearer demo-token-for-testing" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input_dim": 4,
       "output_dim": 2,
       "hidden_layers": [64, 32],
       "ensemble_size": 5,
       "use_mc_dropout": true,
       "heteroscedastic": true
     }' \\
     http://localhost:8000/api/v1/nextgen-ml/bayesian/sessions
```

#### Graph Neural Networks
```bash
curl -X POST -H "Authorization: Bearer demo-token-for-testing" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input_dim": 6,
       "output_dim": 3,
       "conv_type": "gat",
       "use_geometric_attention": true,
       "task_type": "node_prediction"
     }' \\
     http://localhost:8000/api/v1/nextgen-ml/graph/sessions
```

#### Transformer Models
```bash
curl -X POST -H "Authorization: Bearer demo-token-for-testing" \\
     -H "Content-Type: application/json" \\
     -d '{
       "input_dim": 6,
       "output_dim": 3,
       "d_model": 256,
       "nhead": 8,
       "transformer_type": "optimization",
       "use_feature_attention": true
     }' \\
     http://localhost:8000/api/v1/nextgen-ml/transformer/sessions
```

## 🎯 Model Applications

### 1. **Bayesian Neural Networks**
- **Aerospace**: Airfoil optimization with uncertainty
- **Structural**: Material property prediction with confidence intervals
- **Thermal**: Heat exchanger design with aleatory uncertainty

**Key Features**:
- Uncertainty quantification (epistemic + aleatoric)
- Monte Carlo dropout
- Ensemble methods
- Calibrated confidence intervals

### 2. **Graph Neural Networks**
- **FEA**: Finite element mesh analysis
- **CAD**: Geometric feature extraction
- **Topology**: Structural optimization

**Key Features**:
- Geometric deep learning
- Graph attention mechanisms
- Node/edge/graph-level predictions
- Topology-aware modeling

### 3. **Transformer Models**
- **Optimization**: Sequential decision making
- **Time Series**: Performance prediction over time
- **Multi-modal**: Fusion of simulation + experimental data

**Key Features**:
- Self-attention mechanisms
- Feature importance analysis
- Transfer learning capabilities
- Sequential pattern recognition

## 📊 Data Characteristics

| Dataset | Samples | Features | Targets | Domain | Complexity |
|---------|---------|----------|---------|---------|------------|
| Airfoil | 30-200 | 4 | 2 | Aerospace | Medium |
| Structural | 25-150 | 4-5 | 2 | Mechanical | Low |
| Heat Transfer | 25-180 | 4 | 2 | Thermal | Medium |
| Mesh Nodes | 50-100 | 6 | 3 | FEA | High |
| Optimization | 50-100 | 6 | 3 | Multi-objective | High |

## 🔬 Testing Scenarios

### 1. **Uncertainty Quantification**
- Test Bayesian models with airfoil data
- Compare epistemic vs aleatoric uncertainty
- Validate confidence interval calibration

### 2. **Geometric Learning**
- Test Graph NNs with mesh data
- Evaluate attention mechanism effectiveness
- Compare different convolution types (GCN, GAT, SAGE)

### 3. **Sequential Optimization**
- Test Transformers with optimization sequences
- Evaluate convergence prediction accuracy
- Test multi-objective trade-off learning

### 4. **Cross-Domain Transfer**
- Train on one dataset, test on another
- Evaluate domain adaptation capabilities
- Test model generalization

## 🛠️ Utilities

### Files Included:

1. **`test_api_endpoints.sh`**: Comprehensive API testing script
2. **`generate_synthetic_data.py`**: Python script for generating extended datasets
3. **`demo_configurations.json`**: Pre-configured model setups for different scenarios
4. **`dataset_summary.json`**: Generated statistics and metadata

### Dependencies:

For Python data generation:
```bash
pip install numpy pandas
```

For API testing:
```bash
# Requires jq for JSON parsing
brew install jq  # macOS
# or
apt-get install jq  # Ubuntu
```

## 🎨 Visualization

The platform includes built-in visualization for:
- Uncertainty intervals (Bayesian models)
- Attention maps (Graph NNs and Transformers)
- Convergence plots (Optimization sequences)
- Feature importance (All model types)

## 📈 Performance Benchmarks

Expected performance on test datasets:

| Model Type | Dataset | Training Time | Inference Time | Accuracy |
|------------|---------|---------------|----------------|----------|
| Bayesian NN | Airfoil | ~30s | ~10ms | R² > 0.95 |
| Graph NN | Mesh | ~45s | ~15ms | MAE < 5% |
| Transformer | Optimization | ~60s | ~20ms | MSE < 0.1 |

## 🔍 Troubleshooting

### Common Issues:

1. **API Connection**: Ensure both backend (port 8000) and frontend (port 3000) are running
2. **Authentication**: Use the demo token: `demo-token-for-testing`
3. **Data Format**: Ensure CSV files have proper headers and numeric data
4. **Memory**: Large datasets may require adjusting batch sizes

### Debug Commands:

```bash
# Check backend status
curl http://localhost:8000/health

# Check frontend proxy
curl http://localhost:3000/api/v1/nextgen-ml/capabilities

# Test authentication
curl -H "Authorization: Bearer demo-token-for-testing" \\
     http://localhost:8000/api/v1/nextgen-ml/capabilities
```

## 🎯 Next Steps

1. **Upload Custom Data**: Use the dataset upload functionality in the web interface
2. **Model Training**: Train models on your custom datasets
3. **Hyperparameter Tuning**: Experiment with different configurations
4. **Production Deployment**: Scale models for production use
5. **Integration**: Connect with existing engineering workflows

## 📞 Support

For issues or questions:
- Check the main project README
- Review API documentation at http://localhost:8000/docs
- Test with provided sample data first
- Verify all services are running correctly

---

**🚀 Ready to explore next-generation ML for engineering applications!**