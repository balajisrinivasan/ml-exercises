from pathlib import Path

# project root = two levels up from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# dataset-specific
HOUSING_TARBALL = RAW_DATA_DIR / "housing.tgz"
HOUSING_RAW_DIR = RAW_DATA_DIR / "housing"
HOUSING_CSV = HOUSING_RAW_DIR / "housing.csv"
CALIFORNIA_IMAGE = RAW_DATA_DIR / "my_california.png"