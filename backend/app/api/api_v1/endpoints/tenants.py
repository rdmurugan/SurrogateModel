from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.deps import get_db, get_current_tenant, require_role
from app.models.tenant import Tenant
from app.models.user import User

router = APIRouter()


@router.get("/current")
async def get_current_tenant_info(
    current_tenant: Tenant = Depends(get_current_tenant)
):
    return {
        "id": current_tenant.id,
        "name": current_tenant.name,
        "subdomain": current_tenant.subdomain,
        "subscription_tier": current_tenant.subscription_tier,
        "limits": {
            "max_users": current_tenant.max_users,
            "max_models": current_tenant.max_models,
            "max_predictions_per_month": current_tenant.max_predictions_per_month,
            "max_storage_gb": current_tenant.max_storage_gb
        }
    }


@router.get("/usage")
async def get_tenant_usage(
    current_tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    # Count current usage
    users_count = db.query(User).filter(User.tenant_id == current_tenant.id).count()
    models_count = len(current_tenant.models)

    return {
        "users": {
            "current": users_count,
            "limit": current_tenant.max_users
        },
        "models": {
            "current": models_count,
            "limit": current_tenant.max_models
        },
        "storage_gb": {
            "current": 0,  # TODO: Calculate actual storage usage
            "limit": current_tenant.max_storage_gb
        }
    }