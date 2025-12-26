from pathlib import Path

__all__ = [
    "EXAMPLE_UNLABELLED",
    "EXAMPLE_LABELLED",
    "EXAMPLE_PREDICTIONS",
    "DATASET1_UNLABELLED",
    "DATASET1_LABELLED",
    "DATASET2_UNLABELLED",
    "DATASET2_LABELLED",
    "DATASET3_UNLABELLED",
    "DATASET3_LABELLED",
    "DATASET4_UNLABELLED",
]

DATA_DIR = Path(__file__).parent / "data"

EXAMPLE_UNLABELLED = DATA_DIR / "example.csv.gz"
EXAMPLE_LABELLED = DATA_DIR / "example_truth.csv.gz"
EXAMPLE_PREDICTIONS = DATA_DIR / "example_predictions.csv.gz"
DATASET1_UNLABELLED = DATA_DIR / "dataset1.csv.gz"
DATASET1_LABELLED = DATA_DIR / "dataset1_truth.csv.gz"
DATASET2_UNLABELLED = DATA_DIR / "dataset2.csv.gz"
DATASET2_LABELLED = DATA_DIR / "dataset2_truth.csv.gz"
DATASET3_UNLABELLED = DATA_DIR / "dataset3.csv.gz"
DATASET3_LABELLED = DATA_DIR / "dataset3_truth.csv.gz"
DATASET4_UNLABELLED = DATA_DIR / "dataset4.csv.gz"
