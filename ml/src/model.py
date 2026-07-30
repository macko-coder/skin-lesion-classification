"""EfficientNet-B0 model for HAM10000 classification via transfer learning."""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ml.src.dataset import CLASSES

NUM_CLASSES = len(CLASSES)


def build_model(freeze_backbone: bool = False) -> nn.Module:
    # IMAGENET1K_V1 weights are what dataloaders.py's normalization constants
    # (IMAGENET_MEAN/STD) and 224x224 resize are matched to — keep in sync.
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freezes the pretrained feature extractor, leaving only the (new,
        # randomly-initialized) classifier head trainable. Useful as a first
        # training phase on a small dataset: large early gradients from the
        # untrained head would otherwise wreck the pretrained features before
        # they get a chance to adapt gradually.
        for param in model.features.parameters():
            param.requires_grad = False

    # classifier is Sequential(Dropout(0.2), Linear(1280, 1000)); swap the
    # 1000-way ImageNet head for our 7 lesion classes.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model
