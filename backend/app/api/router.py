from fastapi import APIRouter

from .health import router as health_router
from .analysis import router as analysis_router
from .chat import router as chat_router


api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router)
api_router.include_router(analysis_router)
api_router.include_router(chat_router)