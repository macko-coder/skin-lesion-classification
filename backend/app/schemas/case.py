"""Pydantic schemas for case-history read/list responses."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CaseListItem(BaseModel):
    """Summary shape for GET /cases (list view) — no probabilities breakdown."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    image_path: str
    predicted_class: str
    created_at: datetime


class CaseRead(CaseListItem):
    """Full detail shape for GET /cases/{id}."""

    probabilities: dict[str, float]
