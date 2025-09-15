from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import time
from app.core.deps import get_db, get_current_tenant
from app.models.surrogate_model import SurrogateModel
from app.models.prediction import Prediction
from app.models.tenant import Tenant
from app.ml.training_service import ModelTrainingService

router = APIRouter()


@router.post("/{model_id}/predict")
async def make_prediction(
    model_id: int,
    input_data: Dict[str, Any],
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    # Validate model exists, belongs to tenant, and is deployed
    model = db.query(SurrogateModel).filter(
        SurrogateModel.id == model_id,
        SurrogateModel.tenant_id == current_tenant.id,
        SurrogateModel.is_deployed == True
    ).first()

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Model not found or not deployed"
        )

    # Validate input format
    dataset = model.dataset
    expected_inputs = dataset.input_columns
    provided_inputs = list(input_data.keys())

    missing_inputs = [col for col in expected_inputs if col not in provided_inputs]
    if missing_inputs:
        raise HTTPException(
            status_code=400,
            detail=f"Missing input parameters: {missing_inputs}"
        )

    # Make prediction using trained model
    start_time = time.time()

    try:
        training_service = ModelTrainingService(db)
        prediction_result = await training_service.predict(model_id, input_data)
        prediction_time_ms = (time.time() - start_time) * 1000

        # Extract predictions and uncertainty from result
        output_data = {}
        uncertainty_data = {}

        for target_name, result in prediction_result.items():
            output_data[target_name] = result['prediction']
            uncertainty_data[target_name] = result['uncertainty']

        # Save prediction to database
        prediction = Prediction(
            input_data=input_data,
            output_data=output_data,
            uncertainty_data=uncertainty_data,
            prediction_time_ms=prediction_time_ms,
            model_id=model_id
        )

        db.add(prediction)
        db.commit()

        return {
            "prediction_id": prediction.id,
            "input_data": input_data,
            "output_data": output_data,
            "uncertainty_data": uncertainty_data,
            "prediction_time_ms": prediction_time_ms,
            "model_info": {
                "id": model.id,
                "name": model.name,
                "algorithm": model.algorithm
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/{model_id}/predict/batch")
async def make_batch_prediction(
    model_id: int,
    batch_input_data: List[Dict[str, Any]],
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    # Validate model
    model = db.query(SurrogateModel).filter(
        SurrogateModel.id == model_id,
        SurrogateModel.tenant_id == current_tenant.id,
        SurrogateModel.is_deployed == True
    ).first()

    if not model:
        raise HTTPException(
            status_code=404,
            detail="Model not found or not deployed"
        )

    if len(batch_input_data) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large (max 1000 samples)"
        )

    # Process batch predictions
    results = []
    start_time = time.time()

    for input_data in batch_input_data:
        # Simulate prediction for each input
        output_data = {}
        for output_col in model.dataset.output_columns:
            if output_col in model.dataset.data_statistics:
                stats = model.dataset.data_statistics[output_col]
                output_data[output_col] = round(
                    random.uniform(stats["min"], stats["max"]), 4
                )
            else:
                output_data[output_col] = random.uniform(0, 100)

        results.append({
            "input_data": input_data,
            "output_data": output_data
        })

    total_time_ms = (time.time() - start_time) * 1000

    # Save batch prediction
    prediction = Prediction(
        input_data={"batch": batch_input_data},
        output_data={"batch": results},
        prediction_time_ms=total_time_ms,
        batch_size=len(batch_input_data),
        model_id=model_id
    )

    db.add(prediction)
    db.commit()

    return {
        "prediction_id": prediction.id,
        "results": results,
        "batch_size": len(batch_input_data),
        "total_time_ms": total_time_ms,
        "avg_time_per_sample_ms": total_time_ms / len(batch_input_data)
    }


@router.get("/{model_id}/predictions")
async def get_model_predictions(
    model_id: int,
    limit: int = 100,
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    # Validate model belongs to tenant
    model = db.query(SurrogateModel).filter(
        SurrogateModel.id == model_id,
        SurrogateModel.tenant_id == current_tenant.id
    ).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    predictions = db.query(Prediction).filter(
        Prediction.model_id == model_id
    ).order_by(Prediction.created_at.desc()).limit(limit).all()

    return [
        {
            "id": pred.id,
            "input_data": pred.input_data,
            "output_data": pred.output_data,
            "prediction_time_ms": pred.prediction_time_ms,
            "batch_size": pred.batch_size,
            "created_at": pred.created_at
        }
        for pred in predictions
    ]