"""FastAPI app factory: CORS, lifespan model loading, router registration, static file serving."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import health
from backend.app.core.config import get_settings

settings = get_settings()

app = FastAPI(title="Skin Lesion Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)

# predict/cases routers, the ML-model lifespan hook, and the StaticFiles
# mount for storage/ get added once app/ml/inference.py and
# app/storage/files.py have real implementations (later steps).
