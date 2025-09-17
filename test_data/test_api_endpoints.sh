#!/bin/bash

# Surrogate Model Platform API Test Script
# This script tests all the main API endpoints with sample data

BASE_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
TOKEN="demo-token-for-testing"

echo "🚀 Starting Surrogate Model Platform API Tests"
echo "=================================================="

# Test 1: Health Check
echo "1. Testing health check..."
curl -s "$BASE_URL/health" | jq .
echo ""

# Test 2: Next-Gen ML Capabilities
echo "2. Testing Next-Gen ML capabilities..."
curl -s -H "Authorization: Bearer $TOKEN" "$BASE_URL/api/v1/nextgen-ml/capabilities" | jq .
echo ""

# Test 3: Create Bayesian Neural Network Session
echo "3. Creating Bayesian Neural Network session..."
BAYESIAN_SESSION=$(curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dim": 4,
    "output_dim": 2,
    "hidden_layers": [64, 32],
    "activation": "relu",
    "ensemble_size": 5,
    "use_mc_dropout": true,
    "heteroscedastic": true
  }' \
  "$BASE_URL/api/v1/nextgen-ml/bayesian/sessions")

echo "$BAYESIAN_SESSION" | jq .
BAYESIAN_SESSION_ID=$(echo "$BAYESIAN_SESSION" | jq -r '.session_id')
echo "Bayesian Session ID: $BAYESIAN_SESSION_ID"
echo ""

# Test 4: Create Graph Neural Network Session
echo "4. Creating Graph Neural Network session..."
GRAPH_SESSION=$(curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dim": 3,
    "output_dim": 1,
    "conv_type": "gat",
    "use_geometric_attention": true,
    "task_type": "node_prediction"
  }' \
  "$BASE_URL/api/v1/nextgen-ml/graph/sessions")

echo "$GRAPH_SESSION" | jq .
GRAPH_SESSION_ID=$(echo "$GRAPH_SESSION" | jq -r '.session_id')
echo "Graph Session ID: $GRAPH_SESSION_ID"
echo ""

# Test 5: Create Transformer Session
echo "5. Creating Transformer session..."
TRANSFORMER_SESSION=$(curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dim": 5,
    "output_dim": 2,
    "d_model": 128,
    "nhead": 8,
    "transformer_type": "optimization",
    "use_feature_attention": true
  }' \
  "$BASE_URL/api/v1/nextgen-ml/transformer/sessions")

echo "$TRANSFORMER_SESSION" | jq .
TRANSFORMER_SESSION_ID=$(echo "$TRANSFORMER_SESSION" | jq -r '.session_id')
echo "Transformer Session ID: $TRANSFORMER_SESSION_ID"
echo ""

# Test 6: Test Frontend Proxy
echo "6. Testing frontend proxy..."
curl -s "$FRONTEND_URL/api/v1/nextgen-ml/capabilities" | jq '.next_generation_ml_capabilities | keys'
echo ""

# Test 7: Upload Sample Dataset (if upload endpoint exists)
echo "7. Testing dataset upload..."
if [ -f "airfoil_performance.csv" ]; then
  echo "Uploading airfoil performance dataset..."
  # curl -s -X POST \
  #   -H "Authorization: Bearer $TOKEN" \
  #   -F "file=@airfoil_performance.csv" \
  #   -F "name=Airfoil Performance Test" \
  #   -F "description=Test dataset for airfoil aerodynamic analysis" \
  #   "$BASE_URL/api/v1/datasets/upload"
  echo "Dataset upload endpoint not yet implemented - sample data available in CSV files"
else
  echo "Sample CSV files not found in current directory"
fi
echo ""

# Test 8: Generate Predictions (mock test)
echo "8. Testing prediction generation..."
echo "Generating mock predictions for engineering scenarios..."
cat << 'EOF'
{
  "airfoil_prediction": {
    "inputs": {"angle_of_attack": 10, "chord_length": 1.0, "Reynolds_number": 1000000},
    "outputs": {"lift_coefficient": 1.12, "drag_coefficient": 0.025},
    "uncertainty": {"lift_std": 0.05, "drag_std": 0.002}
  },
  "structural_prediction": {
    "inputs": {"load_force": 3000, "material_young_modulus": 200000, "cross_section_area": 0.01},
    "outputs": {"displacement": 0.00015, "stress": 300000},
    "uncertainty": {"displacement_std": 0.000005, "stress_std": 5000}
  }
}
EOF
echo ""

# Test 9: Performance Metrics
echo "9. Testing system performance..."
echo "Measuring API response times..."
time curl -s "$BASE_URL/api/v1/nextgen-ml/capabilities" > /dev/null
echo ""

# Test 10: Error Handling
echo "10. Testing error handling..."
echo "Testing with invalid token..."
curl -s -H "Authorization: Bearer invalid-token" "$BASE_URL/api/v1/nextgen-ml/capabilities" | jq .
echo ""

echo "✅ API Tests Completed!"
echo "========================"
echo "Summary:"
echo "- Bayesian Session: $BAYESIAN_SESSION_ID"
echo "- Graph Session: $GRAPH_SESSION_ID"
echo "- Transformer Session: $TRANSFORMER_SESSION_ID"
echo ""
echo "🎯 Next Steps:"
echo "1. View results at: $FRONTEND_URL"
echo "2. Navigate to Next-Gen ML page: $FRONTEND_URL/nextgen-ml"
echo "3. Test sample datasets in test_data/ directory"
echo "4. Explore uncertainty quantification features"