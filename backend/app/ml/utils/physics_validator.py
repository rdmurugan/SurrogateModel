import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PhysicsValidator:
    """
    Comprehensive physics validation system for surrogate models.

    This system validates surrogate model predictions against known physical laws,
    dimensional consistency, and engineering constraints.
    """

    def __init__(self):
        self.validation_rules = []
        self.dimensional_registry = {}
        self.validation_history = []

    def add_validation_rule(self, rule):
        """Add a physics validation rule"""
        self.validation_rules.append(rule)

    def register_dimensions(self, variable_name: str, dimension: str, units: str = None):
        """
        Register dimensional information for a variable.

        Args:
            variable_name: Name of the variable
            dimension: Physical dimension (e.g., 'Length', 'Force', 'Temperature')
            units: Preferred units (e.g., 'm', 'N', 'K')
        """
        self.dimensional_registry[variable_name] = {
            'dimension': dimension,
            'units': units,
            'base_units': self._get_base_units(dimension)
        }

    def _get_base_units(self, dimension: str) -> Dict[str, float]:
        """Get base SI units for a dimension"""
        base_units_map = {
            'Length': {'m': 1.0},
            'Mass': {'kg': 1.0},
            'Time': {'s': 1.0},
            'Temperature': {'K': 1.0},
            'Force': {'m': 1.0, 'kg': 1.0, 's': -2.0},
            'Pressure': {'m': -1.0, 'kg': 1.0, 's': -2.0},
            'Energy': {'m': 2.0, 'kg': 1.0, 's': -2.0},
            'Power': {'m': 2.0, 'kg': 1.0, 's': -3.0},
            'Velocity': {'m': 1.0, 's': -1.0},
            'Acceleration': {'m': 1.0, 's': -2.0},
            'Area': {'m': 2.0},
            'Volume': {'m': 3.0},
            'Density': {'m': -3.0, 'kg': 1.0},
            'Viscosity': {'m': -1.0, 'kg': 1.0, 's': -1.0},
            'Dimensionless': {}
        }

        return base_units_map.get(dimension, {})

    def validate_predictions(self, predictions: Dict[str, Any],
                           inputs: Dict[str, Any],
                           problem_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate predictions against physics constraints.

        Args:
            predictions: Model predictions
            inputs: Input values
            problem_context: Additional context about the problem

        Returns:
            Validation results
        """
        validation_results = {
            'overall_valid': True,
            'validation_score': 1.0,
            'violations': [],
            'warnings': [],
            'dimensional_analysis': {},
            'physics_checks': {}
        }

        # Run dimensional analysis
        dim_results = self._validate_dimensions(predictions, inputs)
        validation_results['dimensional_analysis'] = dim_results

        if not dim_results['dimensionally_consistent']:
            validation_results['overall_valid'] = False
            validation_results['violations'].extend(dim_results['violations'])

        # Run physics validation rules
        for rule in self.validation_rules:
            try:
                rule_result = rule.validate(predictions, inputs, problem_context)
                validation_results['physics_checks'][rule.name] = rule_result

                if not rule_result['valid']:
                    validation_results['overall_valid'] = False
                    validation_results['violations'].extend(rule_result.get('violations', []))

                if 'warnings' in rule_result:
                    validation_results['warnings'].extend(rule_result['warnings'])

            except Exception as e:
                logger.warning(f"Error in physics rule {rule.name}: {e}")
                validation_results['warnings'].append(f"Physics rule {rule.name} failed: {e}")

        # Calculate overall validation score
        validation_results['validation_score'] = self._calculate_validation_score(validation_results)

        # Store in history
        self.validation_history.append(validation_results)

        return validation_results

    def _validate_dimensions(self, predictions: Dict[str, Any],
                           inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dimensional consistency"""
        dim_results = {
            'dimensionally_consistent': True,
            'violations': [],
            'dimensional_equations': [],
            'unit_conversions': {}
        }

        # Check each prediction variable
        for var_name, pred_info in predictions.items():
            if var_name in self.dimensional_registry:
                expected_dim = self.dimensional_registry[var_name]

                # Basic dimensional checks
                pred_value = pred_info.get('prediction', 0)

                # Check for infinite or NaN values
                if not np.isfinite(pred_value):
                    dim_results['violations'].append(
                        f"Variable {var_name} has invalid value: {pred_value}"
                    )
                    dim_results['dimensionally_consistent'] = False

                # Check for unrealistic magnitudes
                magnitude_check = self._check_magnitude_reasonableness(
                    var_name, pred_value, expected_dim
                )

                if not magnitude_check['reasonable']:
                    dim_results['violations'].append(
                        f"Variable {var_name} has unrealistic magnitude: {pred_value} "
                        f"(expected range: {magnitude_check['expected_range']})"
                    )

        return dim_results

    def _check_magnitude_reasonableness(self, var_name: str, value: float,
                                      dimension_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the magnitude of a value is physically reasonable"""
        dimension = dimension_info['dimension']

        # Define reasonable ranges for common engineering quantities
        reasonable_ranges = {
            'Temperature': (0, 5000),  # Kelvin
            'Pressure': (0, 1e12),     # Pa
            'Force': (0, 1e12),        # N
            'Length': (1e-12, 1e6),    # m
            'Velocity': (0, 1e6),      # m/s
            'Density': (0, 50000),     # kg/m³
            'Energy': (0, 1e15),       # J
            'Power': (0, 1e12),        # W
            'Dimensionless': (-1e6, 1e6)
        }

        if dimension in reasonable_ranges:
            min_val, max_val = reasonable_ranges[dimension]
            reasonable = min_val <= abs(value) <= max_val

            return {
                'reasonable': reasonable,
                'expected_range': (min_val, max_val),
                'actual_value': value
            }

        return {'reasonable': True, 'expected_range': None, 'actual_value': value}

    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score (0-1, 1 = fully valid)"""
        base_score = 1.0

        # Penalize violations
        num_violations = len(validation_results['violations'])
        violation_penalty = min(0.2 * num_violations, 0.8)

        # Penalize warnings (less severe)
        num_warnings = len(validation_results['warnings'])
        warning_penalty = min(0.05 * num_warnings, 0.2)

        # Calculate physics check scores
        physics_scores = []
        for check_name, check_result in validation_results['physics_checks'].items():
            if 'score' in check_result:
                physics_scores.append(check_result['score'])
            else:
                physics_scores.append(1.0 if check_result['valid'] else 0.0)

        if physics_scores:
            avg_physics_score = np.mean(physics_scores)
        else:
            avg_physics_score = 1.0

        # Combined score
        final_score = max(0.0, base_score - violation_penalty - warning_penalty) * avg_physics_score

        return final_score


class PhysicsRule(ABC):
    """Abstract base class for physics validation rules"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def validate(self, predictions: Dict[str, Any], inputs: Dict[str, Any],
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate predictions against this physics rule.

        Returns:
            Dictionary with 'valid' (bool), 'score' (float), 'violations' (list), 'warnings' (list)
        """
        pass


class ConservationLawRule(PhysicsRule):
    """Validate conservation laws (mass, energy, momentum)"""

    def __init__(self, conservation_type: str, tolerance: float = 0.1):
        super().__init__(f"conservation_{conservation_type}")
        self.conservation_type = conservation_type
        self.tolerance = tolerance

    def validate(self, predictions: Dict[str, Any], inputs: Dict[str, Any],
                context: Dict[str, Any] = None) -> Dict[str, Any]:

        if self.conservation_type == 'mass':
            return self._validate_mass_conservation(predictions, inputs, context)
        elif self.conservation_type == 'energy':
            return self._validate_energy_conservation(predictions, inputs, context)
        elif self.conservation_type == 'momentum':
            return self._validate_momentum_conservation(predictions, inputs, context)
        else:
            return {'valid': True, 'score': 1.0, 'violations': [], 'warnings': []}

    def _validate_mass_conservation(self, predictions: Dict[str, Any],
                                  inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mass conservation"""
        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check for mass flow conservation in fluid systems
        if 'mass_flow_in' in inputs and 'mass_flow_out' in predictions:
            mass_in = inputs['mass_flow_in']
            mass_out = predictions['mass_flow_out']['prediction']

            relative_error = abs(mass_in - mass_out) / (abs(mass_in) + 1e-8)

            if relative_error > self.tolerance:
                violations.append(
                    f"Mass conservation violated: in={mass_in:.6f}, out={mass_out:.6f}, "
                    f"error={relative_error:.2%}"
                )
                valid = False
                score = max(0.0, 1.0 - relative_error)

        # Check density bounds
        if 'density' in predictions:
            density = predictions['density']['prediction']
            if density <= 0:
                violations.append(f"Non-physical density: {density}")
                valid = False
                score = 0.0

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }

    def _validate_energy_conservation(self, predictions: Dict[str, Any],
                                    inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate energy conservation"""
        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check energy balance
        energy_terms = ['kinetic_energy', 'potential_energy', 'internal_energy']
        total_energy = 0

        for term in energy_terms:
            if term in predictions:
                energy_value = predictions[term]['prediction']
                if energy_value < 0 and term in ['kinetic_energy']:
                    violations.append(f"Negative {term}: {energy_value}")
                    valid = False
                total_energy += energy_value

        # Check for energy bounds
        if 'total_energy' in context:
            expected_energy = context['total_energy']
            relative_error = abs(total_energy - expected_energy) / (abs(expected_energy) + 1e-8)

            if relative_error > self.tolerance:
                warnings.append(
                    f"Energy balance warning: expected={expected_energy:.6f}, "
                    f"calculated={total_energy:.6f}, error={relative_error:.2%}"
                )

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }

    def _validate_momentum_conservation(self, predictions: Dict[str, Any],
                                      inputs: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate momentum conservation"""
        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check for momentum balance
        if 'force' in predictions and 'acceleration' in predictions:
            force = predictions['force']['prediction']
            acceleration = predictions['acceleration']['prediction']

            # F = ma check (if mass is available)
            if 'mass' in inputs:
                mass = inputs['mass']
                expected_force = mass * acceleration
                relative_error = abs(force - expected_force) / (abs(expected_force) + 1e-8)

                if relative_error > self.tolerance:
                    violations.append(
                        f"Newton's 2nd law violated: F={force:.6f}, ma={expected_force:.6f}, "
                        f"error={relative_error:.2%}"
                    )
                    valid = False
                    score = max(0.0, 1.0 - relative_error)

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }


class ThermodynamicsRule(PhysicsRule):
    """Validate thermodynamic constraints"""

    def __init__(self, tolerance: float = 0.1):
        super().__init__("thermodynamics")
        self.tolerance = tolerance

    def validate(self, predictions: Dict[str, Any], inputs: Dict[str, Any],
                context: Dict[str, Any] = None) -> Dict[str, Any]:

        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check temperature bounds
        if 'temperature' in predictions:
            temp = predictions['temperature']['prediction']
            if temp < 0:  # Below absolute zero
                violations.append(f"Temperature below absolute zero: {temp} K")
                valid = False
                score = 0.0

        # Check ideal gas law (if applicable)
        if all(var in predictions for var in ['pressure', 'temperature']) and 'density' in predictions:
            P = predictions['pressure']['prediction']
            T = predictions['temperature']['prediction']
            rho = predictions['density']['prediction']

            # Check P = rho * R * T (assuming air with R ≈ 287 J/kg·K)
            R = context.get('gas_constant', 287.0) if context else 287.0
            expected_pressure = rho * R * T
            relative_error = abs(P - expected_pressure) / (abs(expected_pressure) + 1e-8)

            if relative_error > self.tolerance:
                warnings.append(
                    f"Ideal gas law deviation: P={P:.2f}, ρRT={expected_pressure:.2f}, "
                    f"error={relative_error:.2%}"
                )

        # Check entropy constraints
        if 'entropy' in predictions:
            entropy = predictions['entropy']['prediction']
            # Basic check: entropy should be finite
            if not np.isfinite(entropy):
                violations.append(f"Invalid entropy value: {entropy}")
                valid = False

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }


class FluidMechanicsRule(PhysicsRule):
    """Validate fluid mechanics constraints"""

    def __init__(self, tolerance: float = 0.1):
        super().__init__("fluid_mechanics")
        self.tolerance = tolerance

    def validate(self, predictions: Dict[str, Any], inputs: Dict[str, Any],
                context: Dict[str, Any] = None) -> Dict[str, Any]:

        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check Bernoulli's equation (if applicable)
        if all(var in predictions for var in ['pressure', 'velocity']) and 'density' in predictions:
            P = predictions['pressure']['prediction']
            v = predictions['velocity']['prediction']
            rho = predictions['density']['prediction']

            # Check that velocity is non-negative
            if v < 0:
                warnings.append(f"Negative velocity magnitude: {v}")

            # Calculate dynamic pressure
            dynamic_pressure = 0.5 * rho * v**2

            # Store for potential Bernoulli analysis
            if context and 'reference_pressure' in context:
                ref_pressure = context['reference_pressure']
                total_pressure = P + dynamic_pressure

                # Check energy conservation along streamline
                if 'reference_total_pressure' in context:
                    ref_total = context['reference_total_pressure']
                    relative_error = abs(total_pressure - ref_total) / (abs(ref_total) + 1e-8)

                    if relative_error > self.tolerance:
                        warnings.append(
                            f"Bernoulli equation deviation: total_P={total_pressure:.2f}, "
                            f"ref_total_P={ref_total:.2f}, error={relative_error:.2%}"
                        )

        # Check Reynolds number bounds (if viscosity available)
        if all(var in predictions for var in ['velocity', 'density']) and 'viscosity' in inputs:
            v = predictions['velocity']['prediction']
            rho = predictions['density']['prediction']
            mu = inputs['viscosity']
            L = context.get('characteristic_length', 1.0) if context else 1.0

            Re = rho * v * L / mu

            # Check for reasonable Reynolds number
            if Re < 0:
                violations.append(f"Negative Reynolds number: {Re}")
                valid = False
            elif Re > 1e8:
                warnings.append(f"Very high Reynolds number: {Re:.2e}")

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }


class StructuralMechanicsRule(PhysicsRule):
    """Validate structural mechanics constraints"""

    def __init__(self, tolerance: float = 0.1):
        super().__init__("structural_mechanics")
        self.tolerance = tolerance

    def validate(self, predictions: Dict[str, Any], inputs: Dict[str, Any],
                context: Dict[str, Any] = None) -> Dict[str, Any]:

        violations = []
        warnings = []
        valid = True
        score = 1.0

        # Check stress-strain relationships
        if 'stress' in predictions and 'strain' in predictions:
            stress = predictions['stress']['prediction']
            strain = predictions['strain']['prediction']

            # Check for reasonable elastic modulus
            if strain != 0:
                E = stress / strain

                # Typical range for engineering materials (Pa)
                if E < 1e6 or E > 1e12:
                    warnings.append(
                        f"Unusual elastic modulus: E={E:.2e} Pa"
                    )

        # Check equilibrium conditions
        if 'force' in predictions and 'displacement' in predictions:
            force = predictions['force']['prediction']
            displacement = predictions['displacement']['prediction']

            # Check force equilibrium (simplified)
            if abs(force) > 0 and abs(displacement) < 1e-12:
                warnings.append(
                    f"Large force with negligible displacement: F={force:.2e}, δ={displacement:.2e}"
                )

        # Check material limits
        if 'stress' in predictions:
            stress = predictions['stress']['prediction']

            # Check against yield strength (if provided)
            if context and 'yield_strength' in context:
                yield_strength = context['yield_strength']
                safety_factor = yield_strength / abs(stress) if stress != 0 else float('inf')

                if safety_factor < 1.0:
                    violations.append(
                        f"Stress exceeds yield strength: σ={stress:.2e}, σ_y={yield_strength:.2e}"
                    )
                    valid = False
                    score = min(score, safety_factor)
                elif safety_factor < 2.0:
                    warnings.append(
                        f"Low safety factor: {safety_factor:.2f}"
                    )

        return {
            'valid': valid,
            'score': score,
            'violations': violations,
            'warnings': warnings
        }


# Factory function for creating domain-specific validators
def create_physics_validator(domain: str) -> PhysicsValidator:
    """
    Create a physics validator for a specific engineering domain.

    Args:
        domain: Engineering domain ('fluid', 'thermal', 'structural', 'general')

    Returns:
        Configured PhysicsValidator
    """
    validator = PhysicsValidator()

    if domain == 'fluid':
        # Fluid mechanics setup
        validator.add_validation_rule(ConservationLawRule('mass'))
        validator.add_validation_rule(ConservationLawRule('momentum'))
        validator.add_validation_rule(FluidMechanicsRule())

        # Register common fluid variables
        validator.register_dimensions('pressure', 'Pressure', 'Pa')
        validator.register_dimensions('velocity', 'Velocity', 'm/s')
        validator.register_dimensions('density', 'Density', 'kg/m³')
        validator.register_dimensions('viscosity', 'Viscosity', 'Pa·s')

    elif domain == 'thermal':
        # Thermal analysis setup
        validator.add_validation_rule(ConservationLawRule('energy'))
        validator.add_validation_rule(ThermodynamicsRule())

        # Register thermal variables
        validator.register_dimensions('temperature', 'Temperature', 'K')
        validator.register_dimensions('heat_flux', 'Power', 'W')
        validator.register_dimensions('thermal_conductivity', 'Power', 'W/m·K')

    elif domain == 'structural':
        # Structural mechanics setup
        validator.add_validation_rule(ConservationLawRule('momentum'))
        validator.add_validation_rule(StructuralMechanicsRule())

        # Register structural variables
        validator.register_dimensions('stress', 'Pressure', 'Pa')
        validator.register_dimensions('strain', 'Dimensionless', '-')
        validator.register_dimensions('displacement', 'Length', 'm')
        validator.register_dimensions('force', 'Force', 'N')

    elif domain == 'general':
        # General engineering validation
        validator.add_validation_rule(ConservationLawRule('mass'))
        validator.add_validation_rule(ConservationLawRule('energy'))
        validator.add_validation_rule(ThermodynamicsRule())

        # Register common variables
        validator.register_dimensions('temperature', 'Temperature', 'K')
        validator.register_dimensions('pressure', 'Pressure', 'Pa')
        validator.register_dimensions('force', 'Force', 'N')
        validator.register_dimensions('energy', 'Energy', 'J')

    return validator