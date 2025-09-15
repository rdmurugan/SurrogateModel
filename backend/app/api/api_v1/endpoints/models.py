from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from app.core.deps import get_db, get_current_tenant, require_role
from app.models.surrogate_model import SurrogateModel
from app.models.dataset import Dataset
from app.models.tenant import Tenant
from app.ml.factory import SurrogateModelFactory
from app.ml.training_service import ModelTrainingService

router = APIRouter()


@router.get("/")
async def list_models(
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    models = db.query(SurrogateModel).filter(
        SurrogateModel.tenant_id == current_tenant.id
    ).all()

    return [
        {
            "id": model.id,
            "name": model.name,
            "algorithm": model.algorithm,
            "training_status": model.training_status,
            "is_deployed": model.is_deployed,
            "validation_metrics": model.validation_metrics,
            "created_at": model.created_at
        }
        for model in models
    ]


@router.post("/")
async def create_model(
    model_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: Tenant = Depends(get_current_tenant),
    current_user = Depends(require_role("engineer")),
    db: Session = Depends(get_db)
):
    # Validate dataset exists and belongs to tenant
    dataset = db.query(Dataset).filter(
        Dataset.id == model_data["dataset_id"],
        Dataset.tenant_id == current_tenant.id
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate algorithm
    available_algorithms = SurrogateModelFactory.get_available_algorithms()
    if model_data["algorithm"] not in available_algorithms:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithm '{model_data['algorithm']}' not supported. Available: {list(available_algorithms.keys())}"
        )

    # Check tenant model limits
    current_models = db.query(SurrogateModel).filter(
        SurrogateModel.tenant_id == current_tenant.id
    ).count()

    if current_models >= current_tenant.max_models:
        raise HTTPException(
            status_code=403,
            detail=f"Model limit reached ({current_tenant.max_models})"
        )

    # Create model record
    model = SurrogateModel(
        name=model_data["name"],
        description=model_data.get("description"),
        algorithm=model_data["algorithm"],
        hyperparameters=model_data.get("hyperparameters", {}),
        dataset_id=dataset.id,
        tenant_id=current_tenant.id,
        training_status="pending"
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    # Start training in background
    training_service = ModelTrainingService(db)

    # Check if hyperparameter optimization is requested
    use_hpo = model_data.get("use_hyperparameter_optimization", False)
    hpo_config = model_data.get("hyperparameter_optimization_config", {})

    if use_hpo:
        background_tasks.add_task(
            training_service.train_with_hyperparameter_optimization,
            model.id,
            hpo_config
        )
    else:
        background_tasks.add_task(training_service.train_model, model.id)

    return {
        "id": model.id,
        "message": "Model training started",
        "status": "pending",
        "hyperparameter_optimization": use_hpo
    }


# Remove the old train_model_task function as it's replaced by ModelTrainingService


@router.get("/{model_id}")
async def get_model(
    model_id: int,
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    model = db.query(SurrogateModel).filter(
        SurrogateModel.id == model_id,
        SurrogateModel.tenant_id == current_tenant.id
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get detailed model information
    training_service = ModelTrainingService(db)
    model_info = training_service.get_model_info(model_id)

    return model_info


@router.get("/algorithms/available")
async def get_available_algorithms():
    """Get information about available algorithms"""
    return SurrogateModelFactory.get_available_algorithms()


@router.post("/algorithms/recommend")
async def recommend_algorithm(
    dataset_info: Dict[str, Any],
    current_tenant: Tenant = Depends(get_current_tenant)
):
    """Get algorithm recommendation based on dataset characteristics"""
    recommendation = SurrogateModelFactory.recommend_algorithm(dataset_info)
    return recommendation


@router.get("/algorithms/comparison")
async def get_algorithm_comparison():
    """Get algorithm comparison matrix"""
    return SurrogateModelFactory.get_algorithm_comparison()


@router.post("/{model_id}/retrain")
async def retrain_model(
    model_id: int,
    retrain_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: Tenant = Depends(get_current_tenant),
    current_user = Depends(require_role("engineer")),
    db: Session = Depends(get_db)
):
    """Retrain an existing model with new hyperparameters"""
    model = db.query(SurrogateModel).filter(
        SurrogateModel.id == model_id,
        SurrogateModel.tenant_id == current_tenant.id
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Update hyperparameters if provided
    if "hyperparameters" in retrain_config:
        model.hyperparameters = retrain_config["hyperparameters"]

    # Reset training status
    model.training_status = "pending"
    model.is_deployed = False
    db.commit()

    # Start retraining
    training_service = ModelTrainingService(db)
    use_hpo = retrain_config.get("use_hyperparameter_optimization", False)
    hpo_config = retrain_config.get("hyperparameter_optimization_config", {})

    if use_hpo:
        background_tasks.add_task(
            training_service.train_with_hyperparameter_optimization,
            model.id,
            hpo_config
        )
    else:
        background_tasks.add_task(training_service.train_model, model.id)

    return {
        "message": "Model retraining started",
        "model_id": model_id,
        "hyperparameter_optimization": use_hpo
    }