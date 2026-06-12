import numpy as np
from unittest.mock import patch, MagicMock

from src.predict import predict_single


# ==========================================================
# Sample Listing
# ==========================================================

SAMPLE_LISTING = {
    "title": "Software Engineer",
    "salary_range": "90000-130000",
    "company_profile": "We are a tech company.",
    "description": "Build scalable APIs using Python and FastAPI.",
    "requirements": "3+ years Python experience. Strong system design skills.",
    "benefits": "Health, dental, 401k",
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1,
    "employment_type": "Full-time",
    "required_experience": "Mid-Senior level",
    "required_education": "Bachelor's Degree",
    "industry": "Information Technology",
    "function": "Engineering",
    "location": "",
    "department": "",
}


# ==========================================================
# Prediction Structure
# ==========================================================

def test_predict_single_structure():

    mock_pipeline = MagicMock()

    mock_pipeline.predict_proba.return_value = np.array(
        [[0.85, 0.15]]
    )

    result = predict_single(
        SAMPLE_LISTING,
        pipeline=mock_pipeline,
        threshold=0.5,
    )

    assert "probability_fake" in result
    assert "is_fake" in result
    assert "risk_level" in result
    assert "threshold_used" in result

    assert result["risk_level"] in (
        "Low",
        "Medium",
        "High",
    )


# ==========================================================
# Risk Levels
# ==========================================================

def test_risk_levels():

    mock_pipeline = MagicMock()

    test_cases = [
        (0.10, "Low"),
        (0.45, "Medium"),
        (0.80, "High"),
    ]

    for probability, expected_risk in test_cases:

        mock_pipeline.predict_proba.return_value = (
            np.array(
                [[1 - probability, probability]]
            )
        )

        result = predict_single(
            SAMPLE_LISTING,
            pipeline=mock_pipeline,
            threshold=0.5,
        )

        assert (
            result["risk_level"]
            == expected_risk
        )


# ==========================================================
# Fake Classification
# ==========================================================

def test_fake_prediction():

    mock_pipeline = MagicMock()

    mock_pipeline.predict_proba.return_value = np.array(
        [[0.20, 0.80]]
    )

    result = predict_single(
        SAMPLE_LISTING,
        pipeline=mock_pipeline,
        threshold=0.5,
    )

    assert result["is_fake"] is True


def test_real_prediction():

    mock_pipeline = MagicMock()

    mock_pipeline.predict_proba.return_value = np.array(
        [[0.80, 0.20]]
    )

    result = predict_single(
        SAMPLE_LISTING,
        pipeline=mock_pipeline,
        threshold=0.5,
    )

    assert result["is_fake"] is False


# ==========================================================
# Model Loading
# ==========================================================

def test_load_model_called():

    mock_pipeline = MagicMock()

    mock_pipeline.predict_proba.return_value = np.array(
        [[0.20, 0.80]]
    )

    with patch(
        "src.predict.load_model",
        return_value=(mock_pipeline, 0.5),
    ):

        result = predict_single(
            SAMPLE_LISTING
        )

        assert result["is_fake"] is True