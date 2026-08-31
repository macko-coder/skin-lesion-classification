"""FastAPI app factory: CORS, lifespan model loading, router registration, static file serving."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import cases, health, predict
from backend.app.core.config import get_settings
from backend.app.ml.inference import get_model

settings = get_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_model()  # load the checkpoint once at startup, not on the first request
    yield


app = FastAPI(title="Skin Lesion Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(cases.router)

# Plain HTML+JS test page for POST /predict -- NOT the real frontend (that's
# frontend/, a separate React app, deliberately not started yet). Mounted
# under /ui rather than / so it can't collide with future API routes.
# Same-origin as the API, so no CORS wrangling needed for the fetch() calls.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

# Serves uploaded lesion images back out so CaseListItem/CaseRead.image_url
# (see app/schemas/case.py) resolves to a fetchable path. Mounted on the
# uploads/ subdir specifically, not all of storage/ -- gradcam/ isn't
# exposed yet since Case has no gradcam_image_path column until Grad-CAM
# is wired up (see app/models/case.py).
app.mount(
    "/storage/uploads",
    StaticFiles(directory=settings.storage_dir / "uploads"),
    name="uploads",
)
