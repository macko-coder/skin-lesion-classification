"""Pydantic schema for the prediction response (predicted class, probabilities).

gradcam_url deferred to a later step, matching Case.gradcam_image_path
being deferred in app/models/case.py until Grad-CAM is actually wired up.
"""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
