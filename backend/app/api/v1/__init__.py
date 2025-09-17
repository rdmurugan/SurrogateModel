from fastapi import APIRouter
from .active_learning import router as active_learning_router
from .nextgen_ml import router as nextgen_ml_router

api_router = APIRouter()
api_router.include_router(active_learning_router)
api_router.include_router(nextgen_ml_router)