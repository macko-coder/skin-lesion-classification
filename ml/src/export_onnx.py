"""Exports the trained checkpoint to ONNX so the architecture (not just the
weights) can be inspected visually, e.g. in Netron (netron.app).

The checkpoint itself is a raw state_dict -- just a flat bag of weight
tensors, with no record of how they're wired together. ONNX captures the
actual computation graph, which is what a graph viewer needs.
"""

import torch

from ml.src.model import build_model
from ml.src.train import CHECKPOINT_PATH, DEVICE, MODELS_DIR

OUTPUT_PATH = MODELS_DIR / "efficientnet_b0_ham10000.onnx"


def export():
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Shape must match dataloaders.py's INPUT_SIZE (224x224) and channel
    # count (RGB) -- torch.onnx.export traces the graph by actually running
    # this dummy input through the model, so it only needs to be shape-valid.
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        input_names=["image"],
        output_names=["logits"],
    )
    print(f"Saved {OUTPUT_PATH}")
    print("Open it at https://netron.app (drag & drop -- runs entirely in your browser, nothing is uploaded)")


if __name__ == "__main__":
    export()
