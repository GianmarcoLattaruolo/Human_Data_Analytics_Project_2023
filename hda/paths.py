from pathlib import Path

# Project root = two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
ESC_50_DIR = DATASETS_DIR / "ESC-50"
ESC_50_DEPTH_DIR = DATASETS_DIR / "ESC-50-depth"
ESC_10_DEPTH_DIR = DATASETS_DIR / "ESC-10-depth"
METADATA_DIR = DATASETS_DIR / "meta"
