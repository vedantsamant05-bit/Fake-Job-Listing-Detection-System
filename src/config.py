from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# Directories
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")
MODEL_DIR = ROOT_DIR / os.getenv("MODEL_DIR", "models")

# Auto-create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Data
# ==========================================================

RAW_DATA_PATH = DATA_DIR / "fake_job_postings.csv"
TARGET_COLUMN = "fraudulent"

# ==========================================================
# Text Columns
# ==========================================================

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

# ==========================================================
# Structured Categorical Columns
# ==========================================================

CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

# ==========================================================
# Engineered Features
# ==========================================================

ENGINEERED_FEATURES = [
    "salary_anomaly",
    "vagueness_score",
    "missing_field_ratio",
    "req_comp_mismatch",
]

# ==========================================================
# Training Configuration
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ==========================================================
# Class Imbalance Handling
# ==========================================================

SMOTE_SAMPLING_STRATEGY = 0.30

# ==========================================================
# Threshold Tuning
# ==========================================================

THRESHOLD_METRIC = "recall"
THRESHOLD_MIN_PRECISION = 0.40

# ==========================================================
# TF-IDF Configuration
# ==========================================================

TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3

# ==========================================================
# Model Metadata
# ==========================================================

MODEL_NAME = "fake-job-detector"
MODEL_VERSION = "1.0.0"

# ==========================================================
# Logging / Debug
# ==========================================================

DEBUG = False
LOG_LEVEL = "INFO"

# ==========================================================
# Saved Artifacts
# ==========================================================

PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"

THRESHOLD_PATH = MODEL_DIR / "threshold.joblib"

FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"

METRICS_PATH = MODEL_DIR / "metrics.json"

SHAP_VALUES_PATH = MODEL_DIR / "shap_values.joblib"