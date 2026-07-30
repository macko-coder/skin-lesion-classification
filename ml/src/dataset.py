"""PyTorch Dataset for HAM10000, backed by the splits.csv manifest."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPLITS_PATH = DATA_DIR / "splits.csv"

# Fixed order so label indices are stable across splits/runs.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    def __init__(self, split: str, transform=None, splits_path: Path = SPLITS_PATH):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(splits_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = DATA_DIR / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["dx"]]
        return image, label
