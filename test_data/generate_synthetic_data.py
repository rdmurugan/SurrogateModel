#!/usr/bin/env python3
"""
Synthetic Data Generator for Surrogate Model Testing
Generates realistic engineering datasets for different model types
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

def generate_airfoil_data(n_samples=100):
    """Generate synthetic airfoil performance data"""
    np.random.seed(42)

    # Input parameters
    angle_of_attack = np.random.uniform(-5, 20, n_samples)
    chord_length = np.random.uniform(0.5, 2.0, n_samples)
    reynolds_number = np.random.uniform(500000, 2000000, n_samples)
    thickness_ratio = np.random.uniform(0.08, 0.20, n_samples)

    # Physics-based relationships with noise
    lift_coefficient = (
        0.1 * angle_of_attack +
        0.05 * chord_length +
        0.0000003 * reynolds_number -
        2.0 * thickness_ratio +
        0.001 * angle_of_attack**2 +
        np.random.normal(0, 0.05, n_samples)
    )

    drag_coefficient = (
        0.005 +
        0.0001 * angle_of_attack**2 +
        0.002 * thickness_ratio**2 +
        np.random.normal(0, 0.002, n_samples)
    )

    # Ensure physical constraints
    lift_coefficient = np.clip(lift_coefficient, -0.5, 2.0)
    drag_coefficient = np.clip(drag_coefficient, 0.005, 0.2)

    return pd.DataFrame({
        'angle_of_attack': angle_of_attack,
        'chord_length': chord_length,
        'Reynolds_number': reynolds_number,
        'thickness_ratio': thickness_ratio,
        'lift_coefficient': lift_coefficient,
        'drag_coefficient': drag_coefficient
    })

def generate_structural_data(n_samples=100):
    """Generate synthetic structural analysis data"""
    np.random.seed(43)

    # Input parameters
    load_force = np.random.uniform(500, 10000, n_samples)
    young_modulus = np.random.uniform(150000, 250000, n_samples)
    cross_section = np.random.uniform(0.005, 0.025, n_samples)
    length = np.random.uniform(0.5, 3.0, n_samples)
    safety_factor = np.random.uniform(1.5, 4.0, n_samples)

    # Structural mechanics relationships
    displacement = (
        (load_force * length) / (young_modulus * cross_section) +
        np.random.normal(0, 0.00001, n_samples)
    )

    stress = (
        load_force / cross_section +
        np.random.normal(0, 1000, n_samples)
    )

    return pd.DataFrame({
        'load_force': load_force,
        'material_young_modulus': young_modulus,
        'cross_section_area': cross_section,
        'length': length,
        'safety_factor': safety_factor,
        'displacement': displacement,
        'stress': stress
    })

def generate_thermal_data(n_samples=100):
    """Generate synthetic heat transfer data"""
    np.random.seed(44)

    # Input parameters
    temp_inlet = np.random.uniform(250, 600, n_samples)
    flow_rate = np.random.uniform(0.05, 0.3, n_samples)
    thermal_conductivity = np.random.uniform(30, 100, n_samples)
    surface_area = np.random.uniform(0.3, 1.0, n_samples)

    # Heat transfer relationships
    temp_outlet = (
        temp_inlet -
        (flow_rate * thermal_conductivity * surface_area * 0.1) +
        np.random.normal(0, 2, n_samples)
    )

    heat_transfer_rate = (
        thermal_conductivity * surface_area * (temp_inlet - temp_outlet) * flow_rate * 100 +
        np.random.normal(0, 50, n_samples)
    )

    return pd.DataFrame({
        'temperature_inlet': temp_inlet,
        'flow_rate': flow_rate,
        'thermal_conductivity': thermal_conductivity,
        'surface_area': surface_area,
        'temperature_outlet': temp_outlet,
        'heat_transfer_rate': heat_transfer_rate
    })

def generate_mesh_data(n_nodes=50):
    """Generate synthetic mesh data for graph neural networks"""
    np.random.seed(45)

    # Node positions in 3D space
    nodes = []
    for i in range(n_nodes):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)

        # Material and boundary properties
        material_id = np.random.choice([1, 2, 3])  # Different materials
        boundary_condition = np.random.choice([0, 1])  # Fixed or free
        load_magnitude = np.random.uniform(0, 1000) if boundary_condition == 1 else 0

        # Stress calculations (simplified)
        distance_from_center = np.sqrt(x**2 + y**2 + z**2)
        stress_x = load_magnitude * (1 + distance_from_center) * np.random.uniform(0.8, 1.2)
        stress_y = load_magnitude * (1 + distance_from_center) * np.random.uniform(0.8, 1.2)
        stress_z = load_magnitude * (1 + distance_from_center) * np.random.uniform(0.8, 1.2)

        nodes.append({
            'node_id': i,
            'x_coordinate': x,
            'y_coordinate': y,
            'z_coordinate': z,
            'material_id': material_id,
            'boundary_condition': boundary_condition,
            'load_magnitude': load_magnitude,
            'stress_x': stress_x,
            'stress_y': stress_y,
            'stress_z': stress_z
        })

    return pd.DataFrame(nodes)

def generate_optimization_sequence(n_iterations=50):
    """Generate synthetic optimization sequence data"""
    np.random.seed(46)

    iterations = []
    x1, x2 = 0.5, 0.5  # Starting design parameters

    for i in range(n_iterations):
        # Multi-objective optimization problem
        obj1 = (x1 - 0.3)**2 + (x2 - 0.7)**2 + np.random.normal(0, 0.01)
        obj2 = (x1 - 0.8)**2 + (x2 - 0.2)**2 + np.random.normal(0, 0.01)

        # Constraint violation
        constraint = max(0, x1 + x2 - 1.5) + max(0, -x1) + max(0, -x2)

        # Convergence probability (decreases over iterations)
        convergence_prob = max(0.1, 1.0 - i/n_iterations)

        iterations.append({
            'iteration': i,
            'design_parameter_1': x1,
            'design_parameter_2': x2,
            'objective_1': obj1,
            'objective_2': obj2,
            'constraint_violation': constraint,
            'convergence_probability': convergence_prob
        })

        # Update parameters for next iteration (simplified optimizer)
        grad1 = 2 * (x1 - 0.3) + 2 * (x1 - 0.8) + np.random.normal(0, 0.1)
        grad2 = 2 * (x2 - 0.7) + 2 * (x2 - 0.2) + np.random.normal(0, 0.1)

        x1 = np.clip(x1 - 0.1 * grad1, 0, 1)
        x2 = np.clip(x2 - 0.1 * grad2, 0, 1)

    return pd.DataFrame(iterations)

def main():
    """Generate all synthetic datasets"""
    print("🔧 Generating synthetic test datasets...")

    # Create output directory
    output_dir = Path(".")

    # Generate datasets
    datasets = {
        "airfoil_performance_extended": generate_airfoil_data(200),
        "structural_analysis_extended": generate_structural_data(150),
        "heat_transfer_extended": generate_thermal_data(180),
        "mesh_nodes": generate_mesh_data(100),
        "optimization_sequence": generate_optimization_sequence(100)
    }

    # Save datasets
    for name, data in datasets.items():
        filename = output_dir / f"{name}.csv"
        data.to_csv(filename, index=False)
        print(f"✅ Generated {filename} ({len(data)} samples)")

    # Generate summary statistics
    summary = {}
    for name, data in datasets.items():
        summary[name] = {
            "shape": data.shape,
            "columns": list(data.columns),
            "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
            "sample_statistics": data.describe().to_dict()
        }

    # Save summary
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📊 Dataset Summary:")
    for name, info in summary.items():
        print(f"  {name}: {info['shape'][0]} samples, {info['shape'][1]} features")

    print(f"\n🎯 Test Data Generation Complete!")
    print(f"📁 Files saved in: {output_dir.absolute()}")

    return datasets

if __name__ == "__main__":
    datasets = main()