import pandas as pd
from src.config import RAW_DATA_PATH, TARGET_COLUMN, TEXT_COLUMNS, CATEGORICAL_COLUMNS

EXPECTED_COLUMNS = [
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "fraudulent"
]

def load_raw_data(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path)
    _validate(df)
    return df

def _validate(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Target column contains nulls")

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        df[col] = df[col].fillna(0).astype(int)

    df["salary_range"] = df["salary_range"].fillna("").astype(str)

    return df