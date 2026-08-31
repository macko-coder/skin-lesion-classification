"""Pydantic schemas for case-history read/list responses."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


class CaseListItem(BaseModel):
    """Summary shape for GET /cases (list view) — no probabilities breakdown."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_path: str
    predicted_class: str
    created_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def image_url(self) -> str:
        # image_path is stored relative to STORAGE_DIR (e.g. "uploads/<uuid>.jpg");
        # the /storage/uploads mount in main.py serves that same subtree, so this
        # is just the request path the frontend can fetch/<img src> directly.
        # No gradcam_url yet -- Case has no gradcam_image_path column until
        # Grad-CAM is wired up (see app/models/case.py).
        return f"/storage/{self.image_path}"


class CaseRead(CaseListItem):
    """Full detail shape for GET /cases/{id}."""

    probabilities: dict[str, float]
