"""Splits HAM10000 into train/val/test, grouped by lesion_id to avoid data leakage.

Images of the same lesion_id can be near-duplicates (multiple shots of the same
lesion), so the split happens at the lesion level, then all of a lesion's images
inherit its split. Stratified by dx to keep class ratios consistent across splits.

Writes ml/data/splits.csv with one row per image plus a "split" column.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_PATH = DATA_DIR / "splits.csv"
IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remaining fraction goes to test
RANDOM_SEED = 42


def find_image_path(image_id: str) -> str:
    for image_dir in IMAGE_DIRS:
        candidate = DATA_DIR / image_dir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate.relative_to(DATA_DIR))
    raise FileNotFoundError(f"Image file not found for image_id={image_id}")


def main() -> None:
    df = pd.read_csv(METADATA_PATH)

    lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    train_lesions, rest_lesions = train_test_split(
        lesions,
        train_size=TRAIN_FRAC,
        stratify=lesions["dx"],
        random_state=RANDOM_SEED,
    )
    val_size = VAL_FRAC / (1 - TRAIN_FRAC)
    val_lesions, test_lesions = train_test_split(
        rest_lesions,
        train_size=val_size,
        stratify=rest_lesions["dx"],
        random_state=RANDOM_SEED,
    )

    split_by_lesion = {
        **{lid: "train" for lid in train_lesions["lesion_id"]},
        **{lid: "val" for lid in val_lesions["lesion_id"]},
        **{lid: "test" for lid in test_lesions["lesion_id"]},
    }

    df["split"] = df["lesion_id"].map(split_by_lesion)
    df["image_path"] = df["image_id"].apply(find_image_path)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH}")
    print("Images per split:")
    print(df["split"].value_counts())
    print("\nLesions per split:")
    print(df.drop_duplicates("lesion_id")["split"].value_counts())
    print("\nClass distribution per split (%):")
    print(
        pd.crosstab(df["dx"], df["split"], normalize="columns").mul(100).round(1)
    )


if __name__ == "__main__":
    main()
