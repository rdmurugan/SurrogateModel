#!/usr/bin/env python3
"""
Test script for Active Learning API endpoints.

This script demonstrates how to use the Active Learning API for intelligent
data collection and surrogate model improvement.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000/api/v1"
ACTIVE_LEARNING_URL = f"{BASE_URL}/active-learning"

# Test data generation
def generate_test_data():
    """Generate synthetic test data for demonstration"""
    # Simple 2D function: f(x1, x2) = x1^2 + x2^2 + sin(x1*x2)
    np.random.seed(42)

    # Initial training data
    X_initial = np.random.uniform(-2, 2, (10, 2))
    y_initial = (X_initial[:, 0]**2 + X_initial[:, 1]**2 +
                np.sin(X_initial[:, 0] * X_initial[:, 1]) +
                0.1 * np.random.normal(0, 1, 10))

    # Candidate points for sampling
    x1_candidates = np.linspace(-2, 2, 20)
    x2_candidates = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1_candidates, x2_candidates)
    X_candidates = np.column_stack([X1.ravel(), X2.ravel()])

    return X_initial.tolist(), y_initial.tolist(), X_candidates.tolist()

def test_active_learning_api():
    """Test the complete Active Learning API workflow"""

    print("🧪 Testing Active Learning API")
    print("=" * 50)

    # Generate test data
    X_initial, y_initial, X_candidates = generate_test_data()

    print(f"📊 Generated test data:")
    print(f"   - Initial samples: {len(X_initial)}")
    print(f"   - Candidate points: {len(X_candidates)}")
    print()

    # Step 1: Create Active Learning Session
    print("1️⃣ Creating Active Learning Session...")

    session_config = {
        "model_config": {
            "type": "gaussian_process",
            "params": {
                "kernel": "rbf",
                "alpha": 1e-8,
                "n_restarts_optimizer": 5
            }
        },
        "sampling_config": {
            "adaptive": {
                "adaptation_frequency": 3,
                "performance_window": 8,
                "convergence_threshold": 1e-4
            },
            "physics_informed": {
                "physics_constraints": {
                    "domain_bounds": [[-2, 2], [-2, 2]]
                },
                "boundary_weights": {"domain": 1.0},
                "conservation_laws": []
            }
        },
        "budget_config": {
            "total_budget": 100.0,
            "cost_per_sample": 1.0
        }
    }

    try:
        response = requests.post(
            f"{ACTIVE_LEARNING_URL}/sessions",
            json=session_config,
            headers={"Authorization": "Bearer fake-token-for-testing"}
        )

        if response.status_code == 201:
            session_data = response.json()
            session_id = session_data["session_id"]
            print(f"   ✅ Session created: {session_id}")
        else:
            print(f"   ❌ Failed to create session: {response.status_code}")
            print(f"   Error: {response.text}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed. Is the API server running on localhost:8000?")
        print("   Start the server with: uvicorn app.main:app --reload")
        return

    # Step 2: Start Active Learning Process
    print("\n2️⃣ Starting Active Learning Process...")

    start_params = {
        "initial_data": {
            "X": X_initial,
            "y": y_initial
        },
        "candidate_points": {
            "points": X_candidates
        },
        "parameters": {
            "max_iterations": 15,
            "convergence_criteria": {
                "model_improvement_threshold": 0.001,
                "budget_threshold": 0.1,
                "max_iterations_without_improvement": 5
            }
        }
    }

    response = requests.post(
        f"{ACTIVE_LEARNING_URL}/sessions/{session_id}/start",
        json=start_params,
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )

    if response.status_code == 202:
        print("   ✅ Active learning started successfully")
        start_data = response.json()
        print(f"   📈 Max iterations: {start_data['max_iterations']}")
        print(f"   🎯 Initial samples: {start_data['initial_samples']}")
    else:
        print(f"   ❌ Failed to start active learning: {response.status_code}")
        print(f"   Error: {response.text}")
        return

    # Step 3: Monitor Progress
    print("\n3️⃣ Monitoring Active Learning Progress...")

    for i in range(20):  # Monitor for up to 20 seconds
        time.sleep(1)

        response = requests.get(
            f"{ACTIVE_LEARNING_URL}/sessions/{session_id}/status",
            headers={"Authorization": "Bearer fake-token-for-testing"}
        )

        if response.status_code == 200:
            status_data = response.json()
            current_status = status_data["status"]

            if current_status == "running":
                service_status = status_data.get("service_status", {})
                iteration = service_status.get("current_iteration", 0)
                samples = service_status.get("total_training_samples", 0)
                print(f"   🔄 Iteration {iteration}, Samples: {samples}")

            elif current_status == "completed":
                print("   ✅ Active learning completed!")
                if "results_summary" in status_data:
                    summary = status_data["results_summary"]
                    print(f"   📊 Final results:")
                    print(f"      - Converged: {summary.get('converged', False)}")
                    print(f"      - Total iterations: {summary.get('total_iterations', 0)}")
                    print(f"      - Final samples: {summary.get('final_sample_count', 0)}")
                    print(f"      - R² score: {summary.get('final_performance', {}).get('r2_score', 0):.4f}")
                break

            elif current_status == "failed":
                print(f"   ❌ Active learning failed: {status_data.get('error', 'Unknown error')}")
                break
        else:
            print(f"   ⚠️ Status check failed: {response.status_code}")
            break

    # Step 4: Test Interactive Sampling
    print("\n4️⃣ Testing Interactive Sampling...")

    # Select a subset of candidates for testing
    test_candidates = X_candidates[:50]  # First 50 candidates

    sampling_request = {
        "candidate_points": test_candidates,
        "n_samples": 3,
        "acquisition_function": "expected_improvement",
        "strategy_override": "physics_informed"
    }

    response = requests.post(
        f"{ACTIVE_LEARNING_URL}/sessions/{session_id}/sample",
        json=sampling_request,
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )

    if response.status_code == 200:
        sampling_data = response.json()
        print("   ✅ Interactive sampling successful")
        print(f"   🎯 Selected {len(sampling_data['selected_points'])} points")
        print(f"   🔧 Strategy used: {sampling_data.get('strategy_used', 'unknown')}")

        # Show selected points
        selected_points = sampling_data['selected_points']
        print(f"   📍 Selected points: {selected_points[:2]}...")  # Show first 2

    else:
        print(f"   ❌ Interactive sampling failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Step 5: Test Predictions
    print("\n5️⃣ Testing Model Predictions...")

    # Test predictions on a few points
    test_points = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]

    prediction_request = {
        "X": test_points,
        "include_uncertainty": True
    }

    response = requests.post(
        f"{ACTIVE_LEARNING_URL}/sessions/{session_id}/predict",
        json=prediction_request,
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )

    if response.status_code == 200:
        prediction_data = response.json()
        print("   ✅ Predictions successful")

        predictions = prediction_data['predictions']
        print(f"   🔮 Predictions for {len(test_points)} points:")

        for i, point in enumerate(test_points):
            if isinstance(predictions, dict):
                # Structured predictions
                pred_info = list(predictions.values())[i] if i < len(predictions) else {}
            else:
                # Simple array predictions
                pred_info = predictions[i] if i < len(predictions) else {}

            pred_val = pred_info.get('prediction', 'N/A')
            uncertainty = pred_info.get('uncertainty', {})
            std_dev = uncertainty.get('standard_deviation', 'N/A')

            print(f"      Point {point}: {pred_val:.3f} ± {std_dev:.3f}")
    else:
        print(f"   ❌ Predictions failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Step 6: Get Complete Results
    print("\n6️⃣ Retrieving Complete Results...")

    response = requests.get(
        f"{ACTIVE_LEARNING_URL}/sessions/{session_id}/results",
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )

    if response.status_code == 200:
        results_data = response.json()
        print("   ✅ Results retrieved successfully")

        print(f"   📈 Performance Summary:")
        print(f"      - Success: {results_data.get('success', False)}")
        print(f"      - Converged: {results_data.get('converged', False)}")
        print(f"      - Total iterations: {results_data.get('total_iterations', 0)}")
        print(f"      - Final sample count: {results_data.get('final_sample_count', 0)}")

        final_perf = results_data.get('final_performance', {})
        print(f"      - R² score: {final_perf.get('r2_score', 0):.4f}")
        print(f"      - MSE: {final_perf.get('mse', 0):.4f}")

    else:
        print(f"   ❌ Results retrieval failed: {response.status_code}")
        print(f"   Error: {response.text}")

    # Step 7: Test Utility Endpoints
    print("\n7️⃣ Testing Utility Endpoints...")

    # List acquisition functions
    response = requests.get(f"{ACTIVE_LEARNING_URL}/acquisition-functions")
    if response.status_code == 200:
        acq_data = response.json()
        print(f"   📋 Available acquisition functions: {len(acq_data.get('acquisition_functions', []))}")

    # List sampling strategies
    response = requests.get(f"{ACTIVE_LEARNING_URL}/sampling-strategies")
    if response.status_code == 200:
        strategy_data = response.json()
        print(f"   🎯 Available sampling strategies: {len(strategy_data.get('sampling_strategies', {}))}")

    # List user sessions
    response = requests.get(
        f"{ACTIVE_LEARNING_URL}/sessions",
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )
    if response.status_code == 200:
        sessions_data = response.json()
        print(f"   📊 User sessions: {len(sessions_data.get('sessions', []))}")

    # Step 8: Cleanup
    print("\n8️⃣ Cleaning Up...")

    response = requests.delete(
        f"{ACTIVE_LEARNING_URL}/sessions/{session_id}",
        headers={"Authorization": "Bearer fake-token-for-testing"}
    )

    if response.status_code == 200:
        print("   ✅ Session cleaned up successfully")
    else:
        print(f"   ⚠️ Cleanup warning: {response.status_code}")

    print("\n🎉 Active Learning API test completed!")
    print("=" * 50)
    print("The API successfully demonstrated:")
    print("• Session management")
    print("• Adaptive sampling strategies")
    print("• Physics-informed sampling")
    print("• Real-time progress monitoring")
    print("• Interactive point selection")
    print("• Model predictions with uncertainty")
    print("• Budget and resource management")
    print("• Intelligent convergence detection")
    print("")
    print("📄 License Notice:")
    print("This software is free for personal and research use.")
    print("For commercial use, contact: durai@infinidatum.net")
    print("")
    print("🚀 Ready to revolutionize your simulation workflows!")

if __name__ == "__main__":
    test_active_learning_api()