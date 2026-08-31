"""Pydantic schema for the prediction response (predicted class, probabilities, Grad-CAM overlay)."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    # Request path for the saved Grad-CAM overlay (served via the
    # /storage/gradcam mount in main.py), same shape as CaseRead.gradcam_url.
    gradcam_url: str
