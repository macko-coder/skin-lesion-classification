"""Transforms and DataLoaders for HAM10000, ready for EfficientNet-B0 transfer learning."""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.src.dataset import HAM10000Dataset

# EfficientNet-B0's ImageNet-pretrained weights expect inputs normalized with
# these exact statistics; using anything else would mismatch what the
# pretrained filters learned to expect and waste the transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def build_transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize(INPUT_SIZE)]

    if train:
        # Dermatoscopic images have no canonical orientation, so flips/rotation
        # are label-preserving. Augmenting also reduces overfitting on the
        # minority classes (df, vasc), which have very few unique lesions.
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def compute_sample_weights(dataset: HAM10000Dataset) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler, inversely proportional to
    class frequency, so rare classes (e.g. df/vasc at ~1% of train) get
    oversampled within each epoch instead of being drowned out by nv (~67%).
    """
    class_counts = dataset.df["dx"].value_counts()
    sample_weights = dataset.df["dx"].map(lambda dx: 1.0 / class_counts[dx])
    return torch.tensor(sample_weights.values, dtype=torch.double)


def get_dataloaders(batch_size: int = 32, num_workers: int = 0):
    # num_workers=0 avoids Windows multiprocessing pitfalls; if you raise it,
    # the calling script needs an `if __name__ == "__main__":` guard.
    train_ds = HAM10000Dataset("train", transform=build_transforms(train=True))
    val_ds = HAM10000Dataset("val", transform=build_transforms(train=False))
    test_ds = HAM10000Dataset("test", transform=build_transforms(train=False))

    sampler = WeightedRandomSampler(
        compute_sample_weights(train_ds), num_samples=len(train_ds), replacement=True
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl
