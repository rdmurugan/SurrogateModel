from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import json
from app.core.deps import get_db, get_current_tenant, require_role
from app.models.dataset import Dataset
from app.models.tenant import Tenant

router = APIRouter()


@router.get("/")
async def list_datasets(
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    datasets = db.query(Dataset).filter(Dataset.tenant_id == current_tenant.id).all()
    return [
        {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "status": dataset.status,
            "num_samples": dataset.num_samples,
            "input_columns": dataset.input_columns,
            "output_columns": dataset.output_columns,
            "created_at": dataset.created_at
        }
        for dataset in datasets
    ]


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    input_columns: str = Form(...),  # JSON string
    output_columns: str = Form(...),  # JSON string
    current_tenant: Tenant = Depends(get_current_tenant),
    current_user = Depends(require_role("engineer")),
    db: Session = Depends(get_db)
):
    # Validate file format
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    # Parse column specifications
    try:
        input_cols = json.loads(input_columns)
        output_cols = json.loads(output_columns)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid column specification format")

    # Read and validate the file
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        else:  # Excel
            df = pd.read_excel(file.file)

        # Validate columns exist
        all_columns = input_cols + output_cols
        missing_columns = [col for col in all_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )

        # Calculate statistics
        data_stats = {}
        for col in all_columns:
            if df[col].dtype in ['int64', 'float64']:
                data_stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

    # Create dataset record
    dataset = Dataset(
        name=name,
        description=description,
        file_path=f"datasets/{current_tenant.id}/{file.filename}",  # TODO: Save to actual storage
        file_size_mb=len(await file.read()) / (1024 * 1024),
        file_format=file.filename.split('.')[-1],
        input_columns=input_cols,
        output_columns=output_cols,
        num_samples=len(df),
        data_statistics=data_stats,
        tenant_id=current_tenant.id,
        status="processed"
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "message": "Dataset uploaded successfully",
        "num_samples": dataset.num_samples,
        "data_statistics": dataset.data_statistics
    }


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.tenant_id == current_tenant.id
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "status": dataset.status,
        "num_samples": dataset.num_samples,
        "input_columns": dataset.input_columns,
        "output_columns": dataset.output_columns,
        "data_statistics": dataset.data_statistics,
        "created_at": dataset.created_at
    }