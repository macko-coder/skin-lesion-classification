"""Loads the trained EfficientNet-B0 checkpoint once (singleton) and runs prediction on an image."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import get_settings
from ml.src.dataloaders import build_transforms
from ml.src.dataset import CLASSES
from ml.src.model import build_model


@lru_cache
def get_model() -> torch.nn.Module:
    # Cached so the checkpoint loads once per process (first call), not on
    # every prediction -- mirrors get_settings()'s @lru_cache in config.py.
    settings = get_settings()
    model = build_model().to(settings.device)
    model.load_state_dict(
        torch.load(settings.model_checkpoint_path, map_location=settings.device)
    )
    model.eval()
    return model


def predict(image: Image.Image) -> dict[str, float]:
    """Runs one PIL image through the model, returns {class_code: probability} for all 7 CLASSES."""
    settings = get_settings()
    model = get_model()
    transform = build_transforms(train=False)

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(settings.device)
    with torch.no_grad():
        probabilities = F.softmax(model(input_tensor), dim=1)[0]

    return {cls: probabilities[i].item() for i, cls in enumerate(CLASSES)}
