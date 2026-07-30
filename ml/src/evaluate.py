"""Evaluates the trained checkpoint on the held-out test set."""

import torch
from sklearn.metrics import classification_report, confusion_matrix

from ml.src.dataloaders import get_dataloaders
from ml.src.dataset import CLASSES
from ml.src.model import build_model
from ml.src.train import CHECKPOINT_PATH, DEVICE


def evaluate():
    _, _, test_dl = get_dataloaders(batch_size=32)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
