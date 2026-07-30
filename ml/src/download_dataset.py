"""Downloads the HAM10000 dataset from Kaggle into ml/data/.

Requires a Kaggle API token at ~/.kaggle/access_token.
"""

from pathlib import Path

import kaggle

DATASET_REF = "kmader/skin-cancer-mnist-ham10000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_REF, path=str(DATA_DIR), unzip=True)
    print(f"Dataset downloaded and extracted to {DATA_DIR}")


if __name__ == "__main__":
    main()
