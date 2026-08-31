"""SQLAlchemy ORM model for a Case (lesion image, prediction, Grad-CAM overlay, metadata).

Minimal first-pass schema: only the columns required for the core
upload -> predict -> store flow. Deferred for a later migration once the
matching features exist: patient_identifier, note (no case-notes UI yet),
model_version.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Path relative to STORAGE_DIR (e.g. "uploads/<uuid>.jpg"), not an absolute
    # path, so STORAGE_DIR can move (e.g. Docker volume) without touching rows.
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    # Nullable: existing rows created before this column predate Grad-CAM
    # wiring, and a future case could in principle fail overlay generation
    # without failing the whole /predict request.
    gradcam_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(10), nullable=False)
    # {class_code: probability} for all 7 CLASSES from ml/src/dataset.py.
    # Generic JSON (not Postgres JSONB) so tests can run against SQLite.
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
