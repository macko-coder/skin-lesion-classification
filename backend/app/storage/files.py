"""save_upload() / save_gradcam_overlay(): path/filename handling under STORAGE_DIR."""

import uuid
from pathlib import Path

from PIL import Image

from backend.app.core.config import get_settings

UPLOADS_SUBDIR = "uploads"
GRADCAM_SUBDIR = "gradcam"


def save_upload(file_bytes: bytes, original_filename: str) -> str:
    """Saves raw image bytes under STORAGE_DIR/uploads/.

    Takes raw bytes rather than a FastAPI UploadFile so this module stays
    HTTP-agnostic (the route handles reading the upload, this just handles
    the filesystem). Returns the path relative to STORAGE_DIR (e.g.
    "uploads/<uuid>.jpg"), which is what gets stored in Case.image_path.
    """
    settings = get_settings()
    # Only the extension is taken from the original filename (not the name
    # itself) -- the fresh uuid filename avoids both collisions and any
    # path-traversal risk from an attacker-controlled filename.
    ext = Path(original_filename).suffix or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"

    uploads_dir = settings.storage_dir / UPLOADS_SUBDIR
    uploads_dir.mkdir(parents=True, exist_ok=True)
    (uploads_dir / filename).write_bytes(file_bytes)

    return f"{UPLOADS_SUBDIR}/{filename}"


def save_gradcam_overlay(overlay: Image.Image, original_filename: str) -> str:
    """Saves a Grad-CAM overlay image under STORAGE_DIR/gradcam/.

    Mirrors save_upload(): fresh uuid filename (not derived from the case's
    upload filename, same collision/path-traversal reasoning), returns the
    path relative to STORAGE_DIR (e.g. "gradcam/<uuid>.jpg") for
    Case.gradcam_image_path.
    """
    ext = Path(original_filename).suffix or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"

    settings = get_settings()
    gradcam_dir = settings.storage_dir / GRADCAM_SUBDIR
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(gradcam_dir / filename)

    return f"{GRADCAM_SUBDIR}/{filename}"
