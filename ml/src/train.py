"""Training loop for HAM10000 EfficientNet-B0 transfer learning."""

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from ml.src.dataloaders import get_dataloaders
from ml.src.model import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_PATH = MODELS_DIR / "efficientnet_b0_ham10000.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# No class weights here: the WeightedRandomSampler in get_dataloaders() already
# rebalances what the model sees during training, so adding loss-level class
# weights too would double-correct for the nv/df/vasc imbalance.
criterion = nn.CrossEntropyLoss()


def run_epoch(model, dataloader, optimizer=None):
    """One pass over dataloader. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(dataloader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    # Macro F1, not accuracy, is the tracked metric: with nv at ~67% of the
    # data, a model that always predicts nv scores ~67% accuracy but 0 recall
    # on every other class. Macro F1 weights all 7 classes equally, catching that.
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def run_training(model, train_dl, val_dl, optimizer, num_epochs, best_val_f1=0.0):
    """Shared epoch loop: trains, validates, and checkpoints on val_f1 improvement.

    best_val_f1 lets fine_tune() seed this with stage 1's score, so stage 2 only
    overwrites the checkpoint if it actually beats the frozen-backbone model.
    """
    for epoch in range(1, num_epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_dl, optimizer)
        val_loss, val_f1 = run_epoch(model, val_dl)

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> new best val_f1={val_f1:.4f}, saved checkpoint")

    return best_val_f1


def train(
    num_epochs: int = 15,
    freeze_backbone: bool = True,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """Stage 1: train only the classifier head on top of frozen ImageNet features."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)
    model = build_model(freeze_backbone=freeze_backbone).to(DEVICE)

    # Only pass trainable params to the optimizer; with freeze_backbone=True
    # that's just the classifier head.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    return run_training(model, train_dl, val_dl, optimizer, num_epochs)


def fine_tune(num_epochs: int = 10, batch_size: int = 32, lr: float = 1e-5):
    """Stage 2: unfreeze the whole network and keep training from stage 1's
    best weights, at a much lower LR than stage 1 (1e-3 -> 1e-5). The low LR
    matters: the backbone's pretrained filters are already good, so large
    updates here would wreck them rather than gently adapt them to skin lesions.
    """
    train_dl, val_dl, _ = get_dataloaders(batch_size=batch_size)

    model = build_model(freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Score the stage-1 checkpoint first so stage 2 only overwrites it if
    # fine-tuning actually improves on it.
    _, stage1_val_f1 = run_epoch(model, val_dl)
    print(f"Stage 1 checkpoint val_f1={stage1_val_f1:.4f} (fine-tuning baseline)")

    return run_training(
        model, train_dl, val_dl, optimizer, num_epochs, best_val_f1=stage1_val_f1
    )


if __name__ == "__main__":
    train()
    fine_tune()
