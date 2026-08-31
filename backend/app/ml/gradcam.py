"""Single-image Grad-CAM overlay generation, built on the primitives in ml/src/gradcam.py."""

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from backend.app.core.config import get_settings
from backend.app.ml.inference import get_model
from ml.src.dataloaders import build_transforms
from ml.src.gradcam import denormalize


def generate_overlay(image: Image.Image) -> Image.Image:
    """Runs Grad-CAM against the model's own top prediction for one image,
    returns the heatmap-over-lesion overlay as a PIL image.

    Overlay is at the model's input resolution (224x224, see
    ml/src/dataloaders.INPUT_SIZE), not the original upload resolution --
    same tradeoff ml/src/gradcam.py's experimental script makes, since the
    heatmap is only meaningful at the resolution the model actually saw.
    """
    settings = get_settings()
    model = get_model()
    transform = build_transforms(train=False)

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(settings.device)

    # model.features[-1] is EfficientNet-B0's last conv block, the deepest
    # layer that still has spatial resolution (before global pooling) --
    # same target as ml/src/gradcam.py's experimental script. Used as a
    # context manager (pytorch_grad_cam's recommended usage) so the forward
    # hooks it registers on the model get cleaned up after each request.
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        # targets=None -> pytorch_grad_cam explains the model's own top
        # prediction, matching what run_inference() reports as predicted_class.
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = denormalize(input_tensor[0])
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return Image.fromarray(overlay)
