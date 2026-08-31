"""save_upload(): path/filename handling under STORAGE_DIR.

save_gradcam_overlay() deferred until Grad-CAM is wired up
(app/ml/gradcam.py), matching Case.gradcam_image_path being deferred in
app/models/case.py.
"""

import uuid
from pathlib import Path

from backend.app.core.config import get_settings

UPLOADS_SUBDIR = "uploads"


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
