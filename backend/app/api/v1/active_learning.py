from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import numpy as np
import asyncio
from datetime import datetime
import uuid

from ...ml.active_learning.service import ActiveLearningService
from ...ml.active_learning.sampling_strategies import AdaptiveSampler, BatchActiveLearning, PhysicsInformedSampler
from ...ml.active_learning.acquisition.factory import AcquisitionFunctionFactory
from ...models.user import User
from ...core.auth import get_current_user
from ...core.database import get_db
from sqlalchemy.orm import Session


router = APIRouter(prefix="/active-learning", tags=["Active Learning"])

# Global storage for active learning sessions (in production, use Redis or database)
active_sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response
class ActiveLearningConfig(BaseModel):
    """Configuration for active learning session"""
    model_config: Dict[str, Any] = Field(..., description="Surrogate model configuration")
    sampling_config: Optional[Dict[str, Any]] = Field(default=None, description="Sampling strategy configuration")
    budget_config: Optional[Dict[str, Any]] = Field(default=None, description="Budget and resource constraints")
    performance_config: Optional[Dict[str, Any]] = Field(default=None, description="Performance monitoring configuration")

    class Config:
        schema_extra = {
            "example": {
                "model_config": {
                    "type": "gaussian_process",
                    "params": {
                        "kernel": "rbf",
                        "alpha": 1e-8,
                        "n_restarts_optimizer": 10
                    }
                },
                "sampling_config": {
                    "adaptive": {
                        "adaptation_frequency": 5,
                        "performance_window": 10,
                        "convergence_threshold": 1e-4
                    },
                    "physics_informed": {
                        "physics_constraints": {},
                        "boundary_weights": {},
                        "conservation_laws": ["mass_conservation"]
                    }
                },
                "budget_config": {
                    "total_budget": 1000.0,
                    "cost_per_sample": 1.0
                }
            }
        }


class InitialData(BaseModel):
    """Initial training data for active learning"""
    X: List[List[float]] = Field(..., description="Input features")
    y: List[float] = Field(..., description="Target values")

    class Config:
        schema_extra = {
            "example": {
                "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "y": [1.5, 2.5, 3.5]
            }
        }


class CandidatePoints(BaseModel):
    """Candidate points for sampling"""
    points: List[List[float]] = Field(..., description="Candidate sampling points")

    class Config:
        schema_extra = {
            "example": {
                "points": [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.0, 5.0]]
            }
        }


class ActiveLearningParameters(BaseModel):
    """Parameters for active learning execution"""
    max_iterations: int = Field(default=50, ge=1, le=1000, description="Maximum AL iterations")
    convergence_criteria: Optional[Dict[str, Any]] = Field(default=None, description="Convergence criteria")

    class Config:
        schema_extra = {
            "example": {
                "max_iterations": 25,
                "convergence_criteria": {
                    "model_improvement_threshold": 0.001,
                    "budget_threshold": 0.1,
                    "max_iterations_without_improvement": 5
                }
            }
        }


class SamplingRequest(BaseModel):
    """Request for sampling new points"""
    candidate_points: List[List[float]] = Field(..., description="Candidate points to sample from")
    n_samples: int = Field(default=1, ge=1, le=50, description="Number of points to sample")
    acquisition_function: Optional[str] = Field(default=None, description="Acquisition function to use")
    strategy_override: Optional[str] = Field(default=None, description="Override adaptive strategy selection")

    class Config:
        schema_extra = {
            "example": {
                "candidate_points": [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]],
                "n_samples": 2,
                "acquisition_function": "expected_improvement",
                "strategy_override": "physics_informed"
            }
        }


class ExperimentalData(BaseModel):
    """New experimental data to add to training"""
    X: List[List[float]] = Field(..., description="Input features from experiments")
    y: List[float] = Field(..., description="Experimental results")

    class Config:
        schema_extra = {
            "example": {
                "X": [[1.5, 2.5], [2.5, 3.5]],
                "y": [2.0, 3.0]
            }
        }


class PredictionRequest(BaseModel):
    """Request for model predictions"""
    X: List[List[float]] = Field(..., description="Input points for prediction")
    include_uncertainty: bool = Field(default=True, description="Include uncertainty estimates")

    class Config:
        schema_extra = {
            "example": {
                "X": [[1.5, 2.5], [2.5, 3.5]],
                "include_uncertainty": True
            }
        }


class SessionResponse(BaseModel):
    """Response containing session information"""
    session_id: str
    status: str
    message: str
    created_at: datetime


class ActiveLearningResults(BaseModel):
    """Results from active learning execution"""
    session_id: str
    success: bool
    converged: bool
    total_iterations: int
    final_sample_count: int
    final_performance: Dict[str, float]
    execution_summary: Dict[str, Any]


@router.post("/sessions", response_model=SessionResponse)
async def create_active_learning_session(
    config: ActiveLearningConfig,
    current_user: User = Depends(get_current_user)
) -> SessionResponse:
    """
    Create a new active learning session.

    This initializes a new active learning service with the specified configuration.
    The session can then be used for iterative sampling and model improvement.
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())

        # Create active learning service
        al_service = ActiveLearningService(
            model_config=config.model_config,
            sampling_config=config.sampling_config,
            budget_config=config.budget_config,
            performance_config=config.performance_config
        )

        # Store session
        active_sessions[session_id] = {
            'service': al_service,
            'user_id': current_user.id,
            'status': 'initialized',
            'created_at': datetime.now(),
            'config': config.dict()
        }

        return SessionResponse(
            session_id=session_id,
            status='initialized',
            message='Active learning session created successfully',
            created_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create active learning session: {str(e)}"
        )


@router.post("/sessions/{session_id}/start")
async def start_active_learning(
    session_id: str,
    initial_data: InitialData,
    candidate_points: CandidatePoints,
    parameters: ActiveLearningParameters,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Start the active learning process for a session.

    This begins the iterative active learning loop using the provided initial data
    and candidate points. The process runs in the background and updates the session status.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    # Check session status
    if session['status'] not in ['initialized', 'completed', 'failed']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start active learning. Session status: {session['status']}"
        )

    try:
        # Convert data to numpy arrays
        X_initial = np.array(initial_data.X)
        y_initial = np.array(initial_data.y)
        candidates = np.array(candidate_points.points)

        # Validate data dimensions
        if len(X_initial) != len(y_initial):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mismatch between X and y dimensions in initial data"
            )

        if X_initial.shape[1] != candidates.shape[1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dimension mismatch between initial data and candidate points"
            )

        # Update session status
        session['status'] = 'running'
        session['started_at'] = datetime.now()

        # Start active learning in background
        background_tasks.add_task(
            run_active_learning_background,
            session_id,
            X_initial,
            y_initial,
            candidates,
            parameters
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "session_id": session_id,
                "status": "running",
                "message": "Active learning started successfully",
                "initial_samples": len(X_initial),
                "candidate_points": len(candidates),
                "max_iterations": parameters.max_iterations
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        session['status'] = 'failed'
        session['error'] = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start active learning: {str(e)}"
        )


async def run_active_learning_background(
    session_id: str,
    X_initial: np.ndarray,
    y_initial: np.ndarray,
    candidates: np.ndarray,
    parameters: ActiveLearningParameters
):
    """Background task for running active learning"""
    session = active_sessions[session_id]
    al_service = session['service']

    try:
        # Prepare initial data
        initial_data_dict = {'X': X_initial, 'y': y_initial}

        # Run active learning
        results = await al_service.start_active_learning(
            initial_data=initial_data_dict,
            candidate_points=candidates,
            max_iterations=parameters.max_iterations,
            convergence_criteria=parameters.convergence_criteria
        )

        # Update session with results
        session['status'] = 'completed'
        session['completed_at'] = datetime.now()
        session['results'] = results

    except Exception as e:
        session['status'] = 'failed'
        session['error'] = str(e)
        session['failed_at'] = datetime.now()


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Get the current status of an active learning session.

    Returns detailed information about the session progress, performance metrics,
    and current state of the active learning process.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    try:
        # Get service status if available
        al_service = session['service']
        service_status = al_service.get_service_status()

        response_data = {
            "session_id": session_id,
            "status": session['status'],
            "created_at": session['created_at'],
            "service_status": service_status
        }

        # Add timing information
        if 'started_at' in session:
            response_data['started_at'] = session['started_at']
        if 'completed_at' in session:
            response_data['completed_at'] = session['completed_at']
        if 'failed_at' in session:
            response_data['failed_at'] = session['failed_at']

        # Add error information if failed
        if session['status'] == 'failed' and 'error' in session:
            response_data['error'] = session['error']

        # Add results if completed
        if session['status'] == 'completed' and 'results' in session:
            response_data['results_summary'] = {
                'converged': session['results']['converged'],
                'total_iterations': session['results']['total_iterations'],
                'final_sample_count': session['results']['final_sample_count'],
                'final_performance': session['results']['final_performance']
            }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )


@router.get("/sessions/{session_id}/results", response_model=ActiveLearningResults)
async def get_active_learning_results(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> ActiveLearningResults:
    """
    Get complete results from a completed active learning session.

    Returns detailed results including performance metrics, sampling history,
    and model improvement trajectory.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    # Check if session is completed
    if session['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session not completed. Current status: {session['status']}"
        )

    if 'results' not in session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Results not available for completed session"
        )

    try:
        results = session['results']

        return ActiveLearningResults(
            session_id=session_id,
            success=results['success'],
            converged=results['converged'],
            total_iterations=results['total_iterations'],
            final_sample_count=results['final_sample_count'],
            final_performance=results['final_performance'],
            execution_summary={
                'training_history': results['training_history'],
                'sampling_history': results['sampling_history'],
                'budget_summary': results['budget_summary'],
                'performance_summary': results['performance_summary'],
                'adaptive_sampling_summary': results['adaptive_sampling_summary']
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve results: {str(e)}"
        )


@router.post("/sessions/{session_id}/sample")
async def sample_new_points(
    session_id: str,
    request: SamplingRequest,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Sample new points using the active learning strategy.

    This endpoint allows for interactive sampling where users can request
    new experimental points based on the current model state.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    # Check if service is available
    if session['status'] not in ['initialized', 'running', 'completed']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot sample from session with status: {session['status']}"
        )

    try:
        al_service = session['service']
        candidates = np.array(request.candidate_points)

        # Get acquisition function if specified
        acquisition_function = None
        if request.acquisition_function:
            acquisition_function = AcquisitionFunctionFactory.create(request.acquisition_function)

        # Use appropriate sampler
        if request.strategy_override:
            sampler = _get_sampler_by_name(request.strategy_override, al_service)
            sampling_result = sampler.sample(
                al_service.primary_model,
                candidates,
                request.n_samples,
                acquisition_function
            )
        else:
            # Use adaptive sampler
            sampling_result = al_service.adaptive_sampler.sample(
                al_service.primary_model,
                candidates,
                request.n_samples,
                acquisition_function
            )

        # Convert numpy arrays to lists for JSON serialization
        response_data = {
            "session_id": session_id,
            "selected_points": sampling_result['selected_points'].tolist(),
            "selected_indices": sampling_result['selected_indices'].tolist(),
            "n_samples": len(sampling_result['selected_points']),
            "strategy_used": sampling_result.get('strategy_type', 'unknown'),
            "acquisition_scores": sampling_result.get('acquisition_scores', []).tolist() if hasattr(sampling_result.get('acquisition_scores', []), 'tolist') else sampling_result.get('acquisition_scores', [])
        }

        # Add strategy-specific information
        if 'adaptive_info' in sampling_result:
            response_data['adaptive_info'] = sampling_result['adaptive_info']

        if 'physics_insights' in sampling_result:
            response_data['physics_insights'] = sampling_result['physics_insights']

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sample new points: {str(e)}"
        )


@router.post("/sessions/{session_id}/add-data")
async def add_experimental_data(
    session_id: str,
    data: ExperimentalData,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Add new experimental data to the active learning session.

    This allows users to incrementally add experimental results and
    retrain the model for improved predictions.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    try:
        al_service = session['service']

        # Convert to numpy arrays
        X_new = np.array(data.X)
        y_new = np.array(data.y)

        # Validate data
        if len(X_new) != len(y_new):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mismatch between X and y dimensions"
            )

        # Update model with new data
        new_data_dict = {'X': X_new, 'y': y_new}
        await al_service._update_models(new_data_dict)

        # Get updated model performance
        if hasattr(al_service.primary_model, 'training_data'):
            X_current, y_current = al_service.primary_model.training_data
            performance = await al_service._evaluate_model_performance_async(X_current, y_current)
        else:
            performance = {"r2_score": 0.0, "n_samples": len(y_new)}

        return JSONResponse(content={
            "session_id": session_id,
            "message": "Experimental data added successfully",
            "new_samples": len(X_new),
            "total_samples": performance.get('n_samples', 0),
            "updated_performance": performance
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add experimental data: {str(e)}"
        )


@router.post("/sessions/{session_id}/predict")
async def get_model_predictions(
    session_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Get predictions from the trained model in the active learning session.

    Returns predictions with uncertainty estimates for the specified input points.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    try:
        al_service = session['service']
        X_pred = np.array(request.X)

        # Get predictions
        prediction_result = await al_service.get_model_predictions(X_pred)

        # Format predictions for JSON response
        predictions = prediction_result['predictions']

        if isinstance(predictions, dict):
            # Structured predictions
            formatted_predictions = {}
            for key, pred in predictions.items():
                if isinstance(pred, dict):
                    formatted_predictions[key] = pred
                else:
                    formatted_predictions[key] = {'prediction': float(pred)}
        else:
            # Simple array predictions
            formatted_predictions = [
                {'prediction': float(pred)} for pred in np.atleast_1d(predictions)
            ]

        response_data = {
            "session_id": session_id,
            "predictions": formatted_predictions,
            "model_info": {
                "model_type": prediction_result['model_type'],
                "training_samples": prediction_result['training_samples']
            },
            "input_points": len(X_pred)
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictions: {str(e)}"
        )


@router.get("/sessions")
async def list_user_sessions(
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    List all active learning sessions for the current user.

    Returns a summary of all sessions owned by the user with their current status.
    """
    try:
        user_sessions = []

        for session_id, session_data in active_sessions.items():
            if session_data['user_id'] == current_user.id:
                session_summary = {
                    "session_id": session_id,
                    "status": session_data['status'],
                    "created_at": session_data['created_at'],
                    "model_type": session_data['config']['model_config'].get('type', 'unknown')
                }

                # Add timing information
                if 'started_at' in session_data:
                    session_summary['started_at'] = session_data['started_at']
                if 'completed_at' in session_data:
                    session_summary['completed_at'] = session_data['completed_at']

                # Add summary stats if available
                if session_data['status'] == 'completed' and 'results' in session_data:
                    results = session_data['results']
                    session_summary['summary'] = {
                        'converged': results.get('converged', False),
                        'total_iterations': results.get('total_iterations', 0),
                        'final_sample_count': results.get('final_sample_count', 0)
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
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Delete an active learning session.

    This will clean up all resources associated with the session.
    """
    # Validate session
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active learning session not found"
        )

    session = active_sessions[session_id]

    # Check user authorization
    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session"
        )

    try:
        # Cleanup service resources
        al_service = session['service']
        await al_service.cleanup()

        # Remove session
        del active_sessions[session_id]

        return JSONResponse(content={
            "session_id": session_id,
            "message": "Session deleted successfully"
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.get("/acquisition-functions")
async def list_acquisition_functions() -> JSONResponse:
    """
    List available acquisition functions.

    Returns information about all available acquisition functions that can be used
    for active learning sampling strategies.
    """
    try:
        available_functions = AcquisitionFunctionFactory.list_available()

        return JSONResponse(content={
            "acquisition_functions": available_functions,
            "total_functions": len(available_functions)
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list acquisition functions: {str(e)}"
        )


@router.get("/sampling-strategies")
async def list_sampling_strategies() -> JSONResponse:
    """
    List available sampling strategies.

    Returns information about all available sampling strategies including
    adaptive, batch, and physics-informed approaches.
    """
    strategies = {
        "adaptive": {
            "name": "Adaptive Sampling",
            "description": "Meta-strategy that dynamically selects optimal sampling approach",
            "parameters": ["adaptation_frequency", "performance_window", "convergence_threshold"]
        },
        "batch": {
            "name": "Batch Active Learning",
            "description": "Parallel sampling for efficient experimental design",
            "parameters": ["batch_size", "strategy", "diversity_weight"]
        },
        "physics_informed": {
            "name": "Physics-Informed Sampling",
            "description": "Domain knowledge guided sampling for engineering applications",
            "parameters": ["physics_constraints", "boundary_weights", "conservation_laws"]
        },
        "uncertainty_based": {
            "name": "Uncertainty-Based Sampling",
            "description": "Traditional acquisition function based sampling",
            "parameters": ["acquisition_function", "acquisition_params"]
        }
    }

    return JSONResponse(content={
        "sampling_strategies": strategies,
        "total_strategies": len(strategies)
    })


def _get_sampler_by_name(strategy_name: str, al_service: ActiveLearningService) -> BaseSampler:
    """Get sampler instance by strategy name"""
    if strategy_name == "adaptive":
        return al_service.adaptive_sampler
    elif strategy_name == "batch":
        return al_service.specialized_samplers.get('batch', BatchActiveLearning())
    elif strategy_name == "physics_informed":
        return al_service.specialized_samplers.get('physics_informed', PhysicsInformedSampler())
    else:
        # Default to adaptive sampler
        return al_service.adaptive_sampler