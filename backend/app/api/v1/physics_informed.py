from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import torch
from datetime import datetime
import uuid

from ...ml.algorithms.physics_informed_nn import (
    PhysicsInformedNN, PINNTrainer, create_engineering_pinn,
    ConservationLaw, BoundaryCondition, DimensionalConsistency
)
from ...ml.utils.physics_validator import create_physics_validator
from ...models.user import User
from ...core.auth import get_current_user

router = APIRouter(prefix="/physics-informed", tags=["Physics-Informed Neural Networks"])

# Global storage for PINN sessions
pinn_sessions: Dict[str, Dict[str, Any]] = {}


class PhysicsConstraintConfig(BaseModel):
    """Configuration for physics constraints"""
    constraint_type: str = Field(..., description="Type of physics constraint")
    parameters: Dict[str, Any] = Field(default={}, description="Constraint-specific parameters")
    weight: float = Field(default=1.0, description="Weight for this constraint in loss function")

    class Config:
        schema_extra = {
            "example": {
                "constraint_type": "conservation_law",
                "parameters": {
                    "law_type": "mass_conservation",
                    "coordinates": ["x", "y"]
                },
                "weight": 1.0
            }
        }


class PINNConfig(BaseModel):
    """Configuration for Physics-Informed Neural Network"""
    input_dim: int = Field(..., description="Number of input features")
    output_dim: int = Field(..., description="Number of output features")
    hidden_layers: List[int] = Field(default=[50, 50, 50], description="Hidden layer sizes")
    activation: str = Field(default="tanh", description="Activation function")
    physics_constraints: List[PhysicsConstraintConfig] = Field(default=[], description="Physics constraints")
    problem_type: Optional[str] = Field(None, description="Engineering problem type for auto-configuration")

    class Config:
        schema_extra = {
            "example": {
                "input_dim": 2,
                "output_dim": 3,
                "hidden_layers": [64, 64, 32],
                "activation": "tanh",
                "physics_constraints": [
                    {
                        "constraint_type": "conservation_law",
                        "parameters": {"law_type": "mass_conservation"},
                        "weight": 1.0
                    },
                    {
                        "constraint_type": "boundary_condition",
                        "parameters": {"condition_type": "dirichlet", "boundary_value": 0.0},
                        "weight": 1.0
                    }
                ],
                "problem_type": "fluid_flow"
            }
        }


class PINNTrainingConfig(BaseModel):
    """Configuration for PINN training"""
    epochs: int = Field(default=1000, description="Number of training epochs")
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    optimizer_type: str = Field(default="adam", description="Optimizer type")
    data_weight: float = Field(default=1.0, description="Weight for data fitting loss")
    physics_weight: float = Field(default=1.0, description="Weight for physics loss")
    batch_size: Optional[int] = Field(None, description="Batch size (None for full batch)")
    validation_split: float = Field(default=0.1, description="Validation data fraction")

    class Config:
        schema_extra = {
            "example": {
                "epochs": 2000,
                "learning_rate": 1e-3,
                "optimizer_type": "adam",
                "data_weight": 1.0,
                "physics_weight": 0.5,
                "validation_split": 0.1
            }
        }


class PINNTrainingData(BaseModel):
    """Training data for PINN"""
    X_data: List[List[float]] = Field(..., description="Training input data")
    y_data: List[List[float]] = Field(..., description="Training output data")
    X_physics: List[List[float]] = Field(..., description="Physics constraint points")

    class Config:
        schema_extra = {
            "example": {
                "X_data": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "y_data": [[1.0, 0.0, 0.0], [0.5, 0.2, 0.1], [0.3, 0.1, 0.2]],
                "X_physics": [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
            }
        }


class PINNPredictionRequest(BaseModel):
    """Request for PINN predictions"""
    X: List[List[float]] = Field(..., description="Input points")
    include_physics_validation: bool = Field(default=True, description="Include physics validation")
    validation_domain: Optional[str] = Field(None, description="Physics validation domain")

    class Config:
        schema_extra = {
            "example": {
                "X": [[0.5, 0.5], [0.7, 0.3]],
                "include_physics_validation": True,
                "validation_domain": "fluid"
            }
        }


class PhysicsValidationRequest(BaseModel):
    """Request for physics validation"""
    predictions: Dict[str, Any] = Field(..., description="Model predictions to validate")
    inputs: Dict[str, Any] = Field(..., description="Input values")
    problem_context: Optional[Dict[str, Any]] = Field(None, description="Problem context")

    class Config:
        schema_extra = {
            "example": {
                "predictions": {
                    "pressure": {"prediction": 101325.0},
                    "velocity": {"prediction": 10.0},
                    "density": {"prediction": 1.225}
                },
                "inputs": {
                    "temperature": 300.0,
                    "mass_flow_in": 0.5
                },
                "problem_context": {
                    "gas_constant": 287.0,
                    "reference_pressure": 101325.0
                }
            }
        }


@router.post("/sessions")
async def create_pinn_session(
    config: PINNConfig,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Create a new Physics-Informed Neural Network session.

    Supports automatic configuration for common engineering problems
    or custom physics constraint specification.
    """
    try:
        session_id = str(uuid.uuid4())

        # Create PINN model
        if config.problem_type:
            # Use predefined engineering configuration
            model, constraints = create_engineering_pinn(
                problem_type=config.problem_type,
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                hidden_layers=config.hidden_layers,
                activation=config.activation
            )
        else:
            # Create custom PINN
            constraints = []

            # Create physics constraints
            for constraint_config in config.physics_constraints:
                constraint = _create_physics_constraint(constraint_config)
                constraints.append(constraint)

            model = PhysicsInformedNN(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                hidden_layers=config.hidden_layers,
                activation=config.activation,
                physics_constraints=constraints
            )

        # Store session
        pinn_sessions[session_id] = {
            'model': model,
            'constraints': constraints,
            'trainer': None,
            'config': config.dict(),
            'user_id': current_user.id,
            'status': 'initialized',
            'created_at': datetime.now()
        }

        return JSONResponse(content={
            "session_id": session_id,
            "status": "initialized",
            "message": "PINN session created successfully",
            "model_info": {
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "hidden_layers": config.hidden_layers,
                "activation": config.activation,
                "num_constraints": len(constraints),
                "problem_type": config.problem_type
            }
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create PINN session: {str(e)}"
        )


@router.post("/sessions/{session_id}/train")
async def train_pinn(
    session_id: str,
    training_data: PINNTrainingData,
    training_config: PINNTrainingConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Train a Physics-Informed Neural Network.

    Combines data fitting loss with physics-based loss terms to ensure
    physical consistency in predictions.
    """
    # Validate session
    if session_id not in pinn_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PINN session not found"
        )

    session = pinn_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] not in ['initialized', 'failed']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot train PINN. Current status: {session['status']}"
        )

    try:
        model = session['model']

        # Create trainer
        trainer = PINNTrainer(
            model=model,
            data_weight=training_config.data_weight,
            physics_weight=training_config.physics_weight,
            learning_rate=training_config.learning_rate,
            optimizer_type=training_config.optimizer_type
        )

        session['trainer'] = trainer
        session['status'] = 'training'
        session['training_started_at'] = datetime.now()

        # Convert training data to numpy
        X_data = np.array(training_data.X_data)
        y_data = np.array(training_data.y_data)
        X_physics = np.array(training_data.X_physics)

        # Start training in background
        background_tasks.add_task(
            _train_pinn_background,
            session_id,
            trainer,
            X_data,
            y_data,
            X_physics,
            training_config
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "session_id": session_id,
                "status": "training",
                "message": "PINN training started",
                "training_info": {
                    "data_samples": len(X_data),
                    "physics_points": len(X_physics),
                    "epochs": training_config.epochs,
                    "data_weight": training_config.data_weight,
                    "physics_weight": training_config.physics_weight
                }
            }
        )

    except Exception as e:
        session['status'] = 'failed'
        session['error'] = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start PINN training: {str(e)}"
        )


async def _train_pinn_background(
    session_id: str,
    trainer: PINNTrainer,
    X_data: np.ndarray,
    y_data: np.ndarray,
    X_physics: np.ndarray,
    training_config: PINNTrainingConfig
):
    """Background task for PINN training"""
    session = pinn_sessions[session_id]

    try:
        # Run training
        training_results = trainer.train(
            X_data=X_data,
            y_data=y_data,
            X_physics=X_physics,
            epochs=training_config.epochs,
            batch_size=training_config.batch_size,
            validation_split=training_config.validation_split
        )

        # Update session with results
        session['status'] = 'trained'
        session['training_completed_at'] = datetime.now()
        session['training_results'] = training_results
        session['training_metrics'] = trainer.get_training_metrics()

    except Exception as e:
        session['status'] = 'failed'
        session['error'] = str(e)
        session['training_failed_at'] = datetime.now()


@router.get("/sessions/{session_id}/training-status")
async def get_training_status(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """Get current training status and metrics"""
    # Validate session
    if session_id not in pinn_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PINN session not found"
        )

    session = pinn_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    try:
        response_data = {
            "session_id": session_id,
            "status": session['status'],
            "created_at": session['created_at']
        }

        # Add training timing
        if 'training_started_at' in session:
            response_data['training_started_at'] = session['training_started_at']
        if 'training_completed_at' in session:
            response_data['training_completed_at'] = session['training_completed_at']
        if 'training_failed_at' in session:
            response_data['training_failed_at'] = session['training_failed_at']

        # Add error if failed
        if session['status'] == 'failed' and 'error' in session:
            response_data['error'] = session['error']

        # Add training results if completed
        if session['status'] == 'trained':
            response_data['training_results'] = session.get('training_results', {})
            response_data['training_metrics'] = session.get('training_metrics', {})

        # Add current training progress if available
        trainer = session.get('trainer')
        if trainer and hasattr(trainer, 'training_history'):
            if trainer.training_history:
                latest_epoch = trainer.training_history[-1]
                response_data['current_progress'] = {
                    'current_epoch': latest_epoch['epoch'],
                    'current_loss': latest_epoch['total_loss'],
                    'data_loss': latest_epoch['data_loss'],
                    'physics_loss': latest_epoch['physics_loss']
                }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )


@router.post("/sessions/{session_id}/predict")
async def predict_with_pinn(
    session_id: str,
    request: PINNPredictionRequest,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Make predictions using trained PINN with optional physics validation.

    Returns predictions that satisfy physics constraints and includes
    validation against known physical laws.
    """
    # Validate session
    if session_id not in pinn_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PINN session not found"
        )

    session = pinn_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] != 'trained':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PINN not trained. Current status: {session['status']}"
        )

    try:
        trainer = session['trainer']
        X = np.array(request.X)

        # Make predictions
        predictions = trainer.predict(X)

        # Physics validation if requested
        validation_results = None
        if request.include_physics_validation:
            validator = create_physics_validator(request.validation_domain or 'general')

            # Convert predictions for validation
            inputs_for_validation = {f"input_{i}": X[0, i] for i in range(X.shape[1])}

            validation_results = validator.validate_predictions(
                predictions=predictions,
                inputs=inputs_for_validation
            )

        response_data = {
            "session_id": session_id,
            "predictions": predictions,
            "input_points": len(X),
            "physics_informed": True,
            "constraint_count": len(session['constraints'])
        }

        if validation_results:
            response_data['physics_validation'] = validation_results

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make predictions: {str(e)}"
        )


@router.post("/validate-physics")
async def validate_physics_predictions(
    request: PhysicsValidationRequest,
    validation_domain: str = "general",
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Validate predictions against physics constraints.

    Standalone physics validation that can be used with any model predictions
    to check for physical consistency.
    """
    try:
        # Create physics validator
        validator = create_physics_validator(validation_domain)

        # Perform validation
        validation_results = validator.validate_predictions(
            predictions=request.predictions,
            inputs=request.inputs,
            problem_context=request.problem_context
        )

        return JSONResponse(content={
            "validation_results": validation_results,
            "validation_domain": validation_domain,
            "validator_rules": len(validator.validation_rules)
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate physics: {str(e)}"
        )


@router.get("/sessions/{session_id}/physics-analysis")
async def get_physics_analysis(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Get detailed physics analysis of the trained PINN.

    Analyzes how well the model satisfies physics constraints and
    identifies potential areas of concern.
    """
    # Validate session
    if session_id not in pinn_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PINN session not found"
        )

    session = pinn_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] != 'trained':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PINN not trained. Current status: {session['status']}"
        )

    try:
        model = session['model']
        trainer = session['trainer']
        constraints = session['constraints']

        analysis = {
            'session_info': {
                'session_id': session_id,
                'created_at': session['created_at'],
                'training_completed_at': session.get('training_completed_at')
            },
            'model_architecture': {
                'input_dim': model.input_dim,
                'output_dim': model.output_dim,
                'hidden_layers': [layer.out_features for layer in model.network if hasattr(layer, 'out_features')],
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'physics_constraints': [],
            'training_metrics': session.get('training_metrics', {}),
            'constraint_analysis': {}
        }

        # Analyze physics constraints
        for i, constraint in enumerate(constraints):
            constraint_info = {
                'index': i,
                'type': type(constraint).__name__,
                'weight': getattr(constraint, 'weight', 1.0)
            }

            # Add constraint-specific information
            if hasattr(constraint, 'law_type'):
                constraint_info['law_type'] = constraint.law_type
            if hasattr(constraint, 'condition_type'):
                constraint_info['condition_type'] = constraint.condition_type

            analysis['physics_constraints'].append(constraint_info)

        # Analyze constraint satisfaction (if training history available)
        if trainer and hasattr(trainer, 'training_history'):
            physics_losses = [h['physics_loss'] for h in trainer.training_history]
            data_losses = [h['data_loss'] for h in trainer.training_history]

            analysis['constraint_analysis'] = {
                'final_physics_loss': physics_losses[-1] if physics_losses else 0.0,
                'final_data_loss': data_losses[-1] if data_losses else 0.0,
                'physics_to_data_ratio': physics_losses[-1] / (data_losses[-1] + 1e-8) if physics_losses and data_losses else 0.0,
                'physics_loss_reduction': (physics_losses[0] - physics_losses[-1]) / (physics_losses[0] + 1e-8) if len(physics_losses) > 1 else 0.0,
                'convergence_assessment': _assess_physics_convergence(trainer.training_history)
            }

        return JSONResponse(content=analysis)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate physics analysis: {str(e)}"
        )


@router.get("/sessions")
async def list_pinn_sessions(
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """List all PINN sessions for the current user"""
    try:
        user_sessions = []

        for session_id, session_data in pinn_sessions.items():
            if session_data['user_id'] == current_user.id:
                session_summary = {
                    "session_id": session_id,
                    "status": session_data['status'],
                    "created_at": session_data['created_at'],
                    "problem_type": session_data['config'].get('problem_type'),
                    "input_dim": session_data['config']['input_dim'],
                    "output_dim": session_data['config']['output_dim'],
                    "constraint_count": len(session_data['constraints'])
                }

                if 'training_completed_at' in session_data:
                    session_summary['training_completed_at'] = session_data['training_completed_at']

                if 'training_metrics' in session_data:
                    metrics = session_data['training_metrics']
                    session_summary['training_summary'] = {
                        'final_loss': metrics.get('final_total_loss'),
                        'converged_epoch': metrics.get('convergence_epoch'),
                        'physics_ratio': metrics.get('physics_vs_data_ratio')
                    }

                user_sessions.append(session_summary)

        return JSONResponse(content={
            "sessions": user_sessions,
            "total_sessions": len(user_sessions)
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def delete_pinn_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """Delete a PINN session"""
    # Validate session
    if session_id not in pinn_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PINN session not found"
        )

    session = pinn_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session"
        )

    try:
        # Clean up resources
        del pinn_sessions[session_id]

        return JSONResponse(content={
            "session_id": session_id,
            "message": "PINN session deleted successfully"
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.get("/problem-types")
async def list_problem_types() -> JSONResponse:
    """List available pre-configured engineering problem types"""
    problem_types = {
        "fluid_flow": {
            "name": "Fluid Flow",
            "description": "Computational fluid dynamics problems",
            "constraints": ["mass_conservation", "momentum_conservation", "boundary_conditions"],
            "typical_inputs": ["x", "y", "z", "time"],
            "typical_outputs": ["pressure", "velocity_x", "velocity_y", "velocity_z"]
        },
        "heat_transfer": {
            "name": "Heat Transfer",
            "description": "Thermal analysis and heat conduction",
            "constraints": ["energy_conservation", "thermal_boundary_conditions"],
            "typical_inputs": ["x", "y", "z", "time"],
            "typical_outputs": ["temperature", "heat_flux"]
        },
        "structural": {
            "name": "Structural Mechanics",
            "description": "Solid mechanics and structural analysis",
            "constraints": ["equilibrium", "boundary_conditions"],
            "typical_inputs": ["x", "y", "z"],
            "typical_outputs": ["displacement_x", "displacement_y", "stress", "strain"]
        }
    }

    return JSONResponse(content={
        "problem_types": problem_types,
        "total_types": len(problem_types)
    })


@router.get("/constraint-types")
async def list_constraint_types() -> JSONResponse:
    """List available physics constraint types"""
    constraint_types = {
        "conservation_law": {
            "name": "Conservation Laws",
            "description": "Physical conservation principles",
            "subtypes": ["mass_conservation", "energy_conservation", "momentum_conservation"],
            "parameters": ["law_type", "coordinates", "weight"]
        },
        "boundary_condition": {
            "name": "Boundary Conditions",
            "description": "Constraints at domain boundaries",
            "subtypes": ["dirichlet", "neumann", "robin"],
            "parameters": ["condition_type", "boundary_value", "boundary_function", "weight"]
        },
        "dimensional_consistency": {
            "name": "Dimensional Consistency",
            "description": "Physical dimensional analysis",
            "parameters": ["expected_dimensions", "weight"]
        }
    }

    return JSONResponse(content={
        "constraint_types": constraint_types,
        "total_types": len(constraint_types)
    })


def _create_physics_constraint(config: PhysicsConstraintConfig):
    """Create a physics constraint from configuration"""
    if config.constraint_type == "conservation_law":
        return ConservationLaw(
            law_type=config.parameters.get('law_type', 'mass_conservation'),
            coordinates=config.parameters.get('coordinates', ['x', 'y']),
            weight=config.weight
        )
    elif config.constraint_type == "boundary_condition":
        return BoundaryCondition(
            condition_type=config.parameters.get('condition_type', 'dirichlet'),
            boundary_value=config.parameters.get('boundary_value', 0.0),
            boundary_function=config.parameters.get('boundary_function'),
            weight=config.weight
        )
    elif config.constraint_type == "dimensional_consistency":
        return DimensionalConsistency(
            expected_dimensions=config.parameters.get('expected_dimensions', {}),
            weight=config.weight
        )
    else:
        raise ValueError(f"Unknown constraint type: {config.constraint_type}")


def _assess_physics_convergence(training_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess convergence of physics constraints"""
    if len(training_history) < 10:
        return {"status": "insufficient_data"}

    physics_losses = [h['physics_loss'] for h in training_history]

    # Check for convergence
    recent_losses = physics_losses[-10:]
    loss_std = np.std(recent_losses)
    loss_mean = np.mean(recent_losses)

    convergence_status = "converged" if loss_std / (loss_mean + 1e-8) < 0.01 else "not_converged"

    return {
        "status": convergence_status,
        "final_physics_loss": physics_losses[-1],
        "loss_variability": loss_std / (loss_mean + 1e-8),
        "total_reduction": (physics_losses[0] - physics_losses[-1]) / (physics_losses[0] + 1e-8)
    }