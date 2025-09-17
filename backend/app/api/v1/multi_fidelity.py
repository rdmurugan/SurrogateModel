from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
import uuid

from ...ml.active_learning.multi_fidelity.co_kriging import CoKrigingModel
from ...ml.active_learning.multi_fidelity.hierarchical_model import HierarchicalMultiFidelityModel
from ...ml.active_learning.multi_fidelity.information_fusion import InformationFusionModel
from ...ml.utils.physics_validator import create_physics_validator
from ...models.user import User
from ...core.auth import get_current_user

router = APIRouter(prefix="/multi-fidelity", tags=["Multi-Fidelity Modeling"])

# Global storage for multi-fidelity sessions
multi_fidelity_sessions: Dict[str, Dict[str, Any]] = {}


class FidelityLevelConfig(BaseModel):
    """Configuration for a single fidelity level"""
    level: int = Field(..., description="Fidelity level (0=lowest, higher=better)")
    cost: float = Field(..., description="Relative cost of evaluation")
    accuracy: float = Field(..., description="Expected accuracy (0.0-1.0)")
    name: Optional[str] = Field(None, description="Human-readable name")

    class Config:
        schema_extra = {
            "example": {
                "level": 1,
                "cost": 1.0,
                "accuracy": 0.8,
                "name": "Medium Fidelity CFD"
            }
        }


class MultiFidelityData(BaseModel):
    """Multi-fidelity training data"""
    fidelity_data: Dict[int, Dict[str, List[List[float]]]] = Field(
        ..., description="Data for each fidelity level"
    )

    class Config:
        schema_extra = {
            "example": {
                "fidelity_data": {
                    0: {
                        "X": [[1.0, 2.0], [2.0, 3.0]],
                        "y": [1.5, 2.5]
                    },
                    1: {
                        "X": [[1.5, 2.5], [2.5, 3.5]],
                        "y": [2.0, 3.0]
                    }
                }
            }
        }


class MultiFidelityConfig(BaseModel):
    """Configuration for multi-fidelity modeling"""
    fidelity_levels: List[FidelityLevelConfig] = Field(..., description="Fidelity level definitions")
    model_type: str = Field(default="co_kriging", description="Multi-fidelity model type")
    model_params: Optional[Dict[str, Any]] = Field(default=None, description="Model-specific parameters")
    fusion_config: Optional[Dict[str, Any]] = Field(default=None, description="Information fusion configuration")

    class Config:
        schema_extra = {
            "example": {
                "fidelity_levels": [
                    {"level": 0, "cost": 1.0, "accuracy": 0.6, "name": "Low Fidelity"},
                    {"level": 1, "cost": 5.0, "accuracy": 0.9, "name": "High Fidelity"}
                ],
                "model_type": "hierarchical",
                "model_params": {
                    "base_kernel": "rbf",
                    "use_input_augmentation": True
                },
                "fusion_config": {
                    "method": "adaptive_weighted",
                    "uncertainty_weighting": True
                }
            }
        }


class MultiFidelityPredictionRequest(BaseModel):
    """Request for multi-fidelity predictions"""
    X: List[List[float]] = Field(..., description="Input points")
    target_fidelity: Optional[int] = Field(None, description="Target fidelity level (None for highest)")
    include_all_fidelities: bool = Field(default=False, description="Include predictions from all fidelities")
    include_fusion: bool = Field(default=True, description="Include information fusion results")

    class Config:
        schema_extra = {
            "example": {
                "X": [[1.5, 2.5], [2.5, 3.5]],
                "target_fidelity": 1,
                "include_all_fidelities": True,
                "include_fusion": True
            }
        }


class FidelityRecommendationRequest(BaseModel):
    """Request for fidelity level recommendation"""
    candidate_points: List[List[float]] = Field(..., description="Candidate evaluation points")
    budget_remaining: Optional[float] = Field(None, description="Remaining evaluation budget")
    optimization_horizon: int = Field(default=10, description="Planning horizon for optimization")

    class Config:
        schema_extra = {
            "example": {
                "candidate_points": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "budget_remaining": 50.0,
                "optimization_horizon": 5
            }
        }


@router.post("/sessions")
async def create_multi_fidelity_session(
    config: MultiFidelityConfig,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Create a new multi-fidelity modeling session.

    Supports different multi-fidelity approaches including Co-Kriging,
    hierarchical modeling, and information fusion.
    """
    try:
        session_id = str(uuid.uuid4())

        # Convert fidelity level configs to dict format
        fidelity_levels = [level.dict() for level in config.fidelity_levels]

        # Create multi-fidelity model
        if config.model_type == "co_kriging":
            model = CoKrigingModel(
                fidelity_levels=fidelity_levels,
                correlation_prior=config.model_params.get('correlation_prior', 0.8) if config.model_params else 0.8
            )
        elif config.model_type == "hierarchical":
            model_params = config.model_params or {}
            model = HierarchicalMultiFidelityModel(
                fidelity_levels=fidelity_levels,
                base_kernel=model_params.get('base_kernel', 'rbf'),
                use_input_augmentation=model_params.get('use_input_augmentation', True),
                correlation_threshold=model_params.get('correlation_threshold', 0.3)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model type: {config.model_type}"
            )

        # Create information fusion model if configured
        fusion_model = None
        if config.fusion_config:
            fusion_config = config.fusion_config
            fusion_model = InformationFusionModel(
                fusion_method=fusion_config.get('method', 'adaptive_weighted'),
                uncertainty_weighting=fusion_config.get('uncertainty_weighting', True),
                locality_radius=fusion_config.get('locality_radius', 0.1)
            )

        # Store session
        multi_fidelity_sessions[session_id] = {
            'model': model,
            'fusion_model': fusion_model,
            'config': config.dict(),
            'user_id': current_user.id,
            'status': 'initialized',
            'created_at': datetime.now()
        }

        return JSONResponse(content={
            "session_id": session_id,
            "status": "initialized",
            "message": "Multi-fidelity session created successfully",
            "model_type": config.model_type,
            "fidelity_levels": len(fidelity_levels),
            "fusion_enabled": fusion_model is not None
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create multi-fidelity session: {str(e)}"
        )


@router.post("/sessions/{session_id}/train")
async def train_multi_fidelity_model(
    session_id: str,
    training_data: MultiFidelityData,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Train a multi-fidelity model with the provided data.

    Trains the model on data from multiple fidelity levels and optionally
    fits information fusion weights.
    """
    # Validate session
    if session_id not in multi_fidelity_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multi-fidelity session not found"
        )

    session = multi_fidelity_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    try:
        model = session['model']
        fusion_model = session['fusion_model']

        # Convert training data to numpy arrays
        multi_fidelity_data = {}
        for fidelity_level, data in training_data.fidelity_data.items():
            X = np.array(data['X'])
            y = np.array(data['y'])
            multi_fidelity_data[int(fidelity_level)] = (X, y)

        # Train multi-fidelity model
        model.fit(multi_fidelity_data)

        # Train fusion model if available
        fusion_weights = None
        if fusion_model:
            fusion_model.fit_fusion_weights(multi_fidelity_data)
            fusion_weights = fusion_model.fidelity_weights

        # Update session status
        session['status'] = 'trained'
        session['trained_at'] = datetime.now()
        session['training_summary'] = {
            'fidelity_levels_trained': list(multi_fidelity_data.keys()),
            'total_samples': sum(len(y) for _, y in multi_fidelity_data.values()),
            'samples_per_fidelity': {
                fid: len(y) for fid, (_, y) in multi_fidelity_data.items()
            }
        }

        # Get model analysis
        analysis = {}
        if hasattr(model, 'get_correlation_analysis'):
            analysis['correlation_analysis'] = model.get_correlation_analysis()
        if hasattr(model, 'get_fidelity_analysis'):
            analysis['fidelity_analysis'] = model.get_fidelity_analysis()
        if hasattr(model, 'get_data_summary'):
            analysis['data_summary'] = model.get_data_summary()

        response_data = {
            "session_id": session_id,
            "status": "trained",
            "message": "Multi-fidelity model trained successfully",
            "training_summary": session['training_summary'],
            "model_analysis": analysis
        }

        if fusion_weights:
            response_data['fusion_weights'] = fusion_weights

        return JSONResponse(content=response_data)

    except Exception as e:
        session['status'] = 'failed'
        session['error'] = str(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train multi-fidelity model: {str(e)}"
        )


@router.post("/sessions/{session_id}/predict")
async def predict_multi_fidelity(
    session_id: str,
    request: MultiFidelityPredictionRequest,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Make predictions using the trained multi-fidelity model.

    Can return predictions at specific fidelity levels or fused predictions
    combining information from multiple fidelities.
    """
    # Validate session
    if session_id not in multi_fidelity_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multi-fidelity session not found"
        )

    session = multi_fidelity_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] != 'trained':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not trained. Current status: {session['status']}"
        )

    try:
        model = session['model']
        fusion_model = session['fusion_model']
        X = np.array(request.X)

        # Get predictions
        predictions = {}

        if request.include_all_fidelities:
            # Get predictions from all fidelity levels
            for fidelity in model.fidelity_levels:
                fid_predictions = model.predict(X, fidelity.level)
                predictions[f"fidelity_{fidelity.level}"] = fid_predictions
        else:
            # Get prediction at target fidelity
            target_fidelity = request.target_fidelity
            fid_predictions = model.predict(X, target_fidelity)
            predictions[f"fidelity_{target_fidelity or 'highest'}"] = fid_predictions

        # Include fusion predictions if requested and available
        fusion_predictions = None
        if request.include_fusion and fusion_model and fusion_model.is_trained:
            # Get individual fidelity predictions for fusion
            individual_predictions = {}
            for fidelity in model.fidelity_levels:
                individual_predictions[fidelity.level] = model.predict(X, fidelity.level)

            # Perform fusion
            fusion_predictions = fusion_model.fuse_predictions(individual_predictions, X)
            predictions['fused'] = fusion_predictions

        # Get model insights
        insights = {}

        # Correlation analysis
        if hasattr(model, 'get_correlation_analysis'):
            insights['correlation_analysis'] = model.get_correlation_analysis()

        # Cost-effectiveness analysis
        if hasattr(model, 'get_cost_effectiveness_analysis'):
            insights['cost_effectiveness'] = model.get_cost_effectiveness_analysis()

        # Fusion quality analysis
        if fusion_model and fusion_predictions:
            fusion_analysis = fusion_model.analyze_fusion_quality(
                {fid.level: model.predict(X, fid.level) for fid in model.fidelity_levels}
            )
            insights['fusion_quality'] = fusion_analysis

        return JSONResponse(content={
            "session_id": session_id,
            "predictions": predictions,
            "model_insights": insights,
            "input_points": len(X)
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to make predictions: {str(e)}"
        )


@router.post("/sessions/{session_id}/recommend-fidelity")
async def recommend_fidelity_allocation(
    session_id: str,
    request: FidelityRecommendationRequest,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Recommend optimal fidelity allocation for new evaluations.

    Uses multi-fidelity optimization to suggest which points should be
    evaluated at which fidelity levels to maximize information gain
    per unit cost.
    """
    # Validate session
    if session_id not in multi_fidelity_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multi-fidelity session not found"
        )

    session = multi_fidelity_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] != 'trained':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not trained. Current status: {session['status']}"
        )

    try:
        model = session['model']
        candidates = np.array(request.candidate_points)

        # Get fidelity allocation recommendations
        if hasattr(model, 'optimize_fidelity_allocation'):
            allocation_plan = model.optimize_fidelity_allocation(
                total_budget=request.budget_remaining or 100.0,
                X_candidates=candidates,
                optimization_horizon=request.optimization_horizon
            )
        else:
            # Fallback: simple cost-effectiveness based recommendation
            allocation_plan = []
            remaining_budget = request.budget_remaining or 100.0

            for i, point in enumerate(candidates[:request.optimization_horizon]):
                if remaining_budget <= 0:
                    break

                # Recommend based on cost-effectiveness
                best_fidelity = None
                best_score = 0

                for fidelity in model.fidelity_levels:
                    if fidelity.cost <= remaining_budget:
                        score = fidelity.accuracy / fidelity.cost
                        if score > best_score:
                            best_score = score
                            best_fidelity = fidelity

                if best_fidelity:
                    allocation_plan.append({
                        'step': len(allocation_plan),
                        'point': point.tolist(),
                        'fidelity_level': best_fidelity.level,
                        'cost': best_fidelity.cost,
                        'expected_benefit': best_score,
                        'reasoning': f'Best cost-effectiveness: {best_score:.3f}'
                    })
                    remaining_budget -= best_fidelity.cost

        # Calculate total cost and expected benefit
        total_cost = sum(step['cost'] for step in allocation_plan)
        total_benefit = sum(step.get('expected_benefit', 0) for step in allocation_plan)

        # Get additional insights
        insights = {
            'cost_breakdown': {},
            'fidelity_distribution': {},
            'efficiency_metrics': {
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'cost_benefit_ratio': total_benefit / (total_cost + 1e-8),
                'budget_utilization': total_cost / (request.budget_remaining or 100.0)
            }
        }

        # Calculate cost breakdown by fidelity
        for fidelity in model.fidelity_levels:
            fidelity_steps = [step for step in allocation_plan if step['fidelity_level'] == fidelity.level]
            insights['cost_breakdown'][fidelity.level] = {
                'count': len(fidelity_steps),
                'total_cost': sum(step['cost'] for step in fidelity_steps),
                'fidelity_name': fidelity.name
            }
            insights['fidelity_distribution'][fidelity.level] = len(fidelity_steps)

        return JSONResponse(content={
            "session_id": session_id,
            "allocation_plan": allocation_plan,
            "insights": insights,
            "recommendations": {
                'total_evaluations': len(allocation_plan),
                'estimated_cost': total_cost,
                'budget_remaining': max(0, (request.budget_remaining or 100.0) - total_cost),
                'efficiency_score': total_benefit / (total_cost + 1e-8)
            }
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate fidelity recommendations: {str(e)}"
        )


@router.get("/sessions/{session_id}/analysis")
async def get_multi_fidelity_analysis(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """
    Get comprehensive analysis of the multi-fidelity model.

    Returns detailed insights about fidelity correlations, model performance,
    and fusion quality metrics.
    """
    # Validate session
    if session_id not in multi_fidelity_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multi-fidelity session not found"
        )

    session = multi_fidelity_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )

    if session['status'] != 'trained':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not trained. Current status: {session['status']}"
        )

    try:
        model = session['model']
        fusion_model = session['fusion_model']

        analysis = {
            'session_info': {
                'session_id': session_id,
                'model_type': session['config']['model_type'],
                'created_at': session['created_at'],
                'trained_at': session.get('trained_at'),
                'training_summary': session.get('training_summary', {})
            },
            'fidelity_analysis': {},
            'model_analysis': {},
            'fusion_analysis': {}
        }

        # Fidelity level analysis
        if hasattr(model, 'get_data_summary'):
            analysis['fidelity_analysis']['data_summary'] = model.get_data_summary()

        if hasattr(model, 'get_cost_effectiveness_analysis'):
            analysis['fidelity_analysis']['cost_effectiveness'] = model.get_cost_effectiveness_analysis()

        if hasattr(model, 'get_fidelity_correlation_matrix'):
            correlation_matrix = model.get_fidelity_correlation_matrix()
            analysis['fidelity_analysis']['correlation_matrix'] = correlation_matrix.tolist()

        # Model-specific analysis
        if hasattr(model, 'get_correlation_analysis'):
            analysis['model_analysis']['correlation_analysis'] = model.get_correlation_analysis()

        if hasattr(model, 'get_fidelity_analysis'):
            analysis['model_analysis']['hierarchical_analysis'] = model.get_fidelity_analysis()

        if hasattr(model, 'cross_validate_fidelities'):
            try:
                cv_results = model.cross_validate_fidelities()
                analysis['model_analysis']['cross_validation'] = cv_results
            except Exception as e:
                analysis['model_analysis']['cross_validation'] = {
                    'error': f"Cross-validation failed: {str(e)}"
                }

        # Fusion analysis
        if fusion_model and fusion_model.is_trained:
            try:
                # Get sample predictions for fusion analysis
                sample_X = np.array([[0.5, 0.5]])  # Sample point for analysis
                individual_predictions = {}
                for fidelity in model.fidelity_levels:
                    individual_predictions[fidelity.level] = model.predict(sample_X, fidelity.level)

                fusion_quality = fusion_model.analyze_fusion_quality(individual_predictions)
                analysis['fusion_analysis'] = fusion_quality

            except Exception as e:
                analysis['fusion_analysis'] = {
                    'error': f"Fusion analysis failed: {str(e)}"
                }

        return JSONResponse(content=analysis)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate analysis: {str(e)}"
        )


@router.get("/sessions")
async def list_multi_fidelity_sessions(
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """List all multi-fidelity sessions for the current user"""
    try:
        user_sessions = []

        for session_id, session_data in multi_fidelity_sessions.items():
            if session_data['user_id'] == current_user.id:
                session_summary = {
                    "session_id": session_id,
                    "status": session_data['status'],
                    "created_at": session_data['created_at'],
                    "model_type": session_data['config']['model_type'],
                    "fidelity_levels": len(session_data['config']['fidelity_levels']),
                    "fusion_enabled": session_data['fusion_model'] is not None
                }

                if 'trained_at' in session_data:
                    session_summary['trained_at'] = session_data['trained_at']

                if 'training_summary' in session_data:
                    session_summary['training_summary'] = session_data['training_summary']

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
async def delete_multi_fidelity_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
) -> JSONResponse:
    """Delete a multi-fidelity session"""
    # Validate session
    if session_id not in multi_fidelity_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multi-fidelity session not found"
        )

    session = multi_fidelity_sessions[session_id]

    if session['user_id'] != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session"
        )

    try:
        # Clean up resources
        del multi_fidelity_sessions[session_id]

        return JSONResponse(content={
            "session_id": session_id,
            "message": "Multi-fidelity session deleted successfully"
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.get("/model-types")
async def list_model_types() -> JSONResponse:
    """List available multi-fidelity model types"""
    model_types = {
        "co_kriging": {
            "name": "Co-Kriging",
            "description": "Gaussian Process based multi-fidelity modeling with correlation modeling",
            "parameters": ["correlation_prior"],
            "best_for": "Strongly correlated fidelity levels"
        },
        "hierarchical": {
            "name": "Hierarchical Multi-Fidelity",
            "description": "Recursive correction modeling for complex fidelity relationships",
            "parameters": ["base_kernel", "use_input_augmentation", "correlation_threshold"],
            "best_for": "Complex non-linear fidelity relationships"
        }
    }

    return JSONResponse(content={
        "model_types": model_types,
        "total_types": len(model_types)
    })