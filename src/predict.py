import pandas as pd
import joblib
from joblib import parallel_backend

from src.config import (
    PIPELINE_PATH,
    THRESHOLD_PATH,
    TEXT_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from src.data_loader import basic_clean


# ==========================================================
# Model Loading
# ==========================================================

def load_model():
    """
    Load trained pipeline and tuned threshold.
    """
    pipeline = joblib.load(PIPELINE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return pipeline, threshold


# ==========================================================
# Helpers
# ==========================================================

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected columns exist for inference.
    Useful when users provide partial job listings.
    """

    required_columns = (
        TEXT_COLUMNS
        + CATEGORICAL_COLUMNS
        + [
            "salary_range",
            "telecommuting",
            "has_company_logo",
            "has_questions",
        ]
    )

    binary_columns = [
        "telecommuting",
        "has_company_logo",
        "has_questions",
    ]

    for col in required_columns:

        if col not in df.columns:

            if col in binary_columns:
                df[col] = 0
            else:
                df[col] = ""

    return df


def _risk_level(probability: float) -> str:
    """
    Convert fraud probability into a human-readable risk level.
    """

    if probability < 0.30:
        return "Low"

    if probability < 0.60:
        return "Medium"

    return "High"


# ==========================================================
# Single Prediction
# ==========================================================

def predict_single(
    listing: dict,
    pipeline=None,
    threshold: float = 0.5,
) -> dict:
    """
    Predict authenticity of a single job listing.

    Returns:
    {
        "probability_fake": float,
        "is_fake": bool,
        "risk_level": str,
        "threshold_used": float
    }
    """

    if pipeline is None:
        pipeline, threshold = load_model()

    df = pd.DataFrame([listing])

    df = _ensure_required_columns(df)
    df = basic_clean(df)

    df = df.drop(
        columns=["fraudulent", "job_id"],
        errors="ignore",
    )

    # Use threading backend to avoid loky process-pool executor being
    # garbage-collected between Streamlit reruns on Windows (joblib 1.5 / sk-learn 1.9).
    with parallel_backend("threading", n_jobs=1):
        probability = pipeline.predict_proba(df)[0, 1]

    is_fake = probability >= threshold

    return {
        "probability_fake": round(float(probability), 4),
        "is_fake": bool(is_fake),
        "risk_level": _risk_level(probability),
        "threshold_used": float(threshold),
    }


# ==========================================================
# Batch Prediction
# ==========================================================

def predict_batch(
    df: pd.DataFrame,
    pipeline=None,
    threshold: float = 0.5,
) -> pd.DataFrame:

    if pipeline is None:
        pipeline, threshold = load_model()

    df_clean = df.copy()

    df_clean = _ensure_required_columns(df_clean)
    df_clean = basic_clean(df_clean)

    df_clean = df_clean.drop(
        columns=["fraudulent", "job_id"],
        errors="ignore",
    )

    with parallel_backend("threading", n_jobs=1):
        probabilities = pipeline.predict_proba(df_clean)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = df.copy()

    result["probability_fake"] = probabilities.round(4)

    result["predicted_fake"] = predictions

    result["risk_level"] = [
        _risk_level(p)
        for p in probabilities
    ]

    return result


# ==========================================================
# Quick Manual Test
# ==========================================================

if __name__ == "__main__":

    sample_listing = {
        "title": "Work From Home - Earn $5000 Weekly",
        "description": (
            "No experience needed. "
            "Work from anywhere. "
            "Unlimited earnings."
        ),
        "requirements": "",
        "benefits": "",
        "salary_range": "",
    }

    result = predict_single(sample_listing)

    print(result)