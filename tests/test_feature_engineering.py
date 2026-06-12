import pandas as pd
import numpy as np

from src.feature_engineering import (
    parse_salary,
    salary_anomaly_score,
    vagueness_score,
    missing_field_ratio,
    req_comp_mismatch,
    StructuredFeatureEngineer,
)


# ==========================================================
# Salary Parsing
# ==========================================================

def test_parse_salary_standard():
    low, high = parse_salary("50000-80000")

    assert low == 50000.0
    assert high == 80000.0


def test_parse_salary_k_format():
    low, high = parse_salary("$50k-$80k")

    assert low == 50000.0
    assert high == 80000.0


def test_parse_salary_missing():
    low, high = parse_salary("")

    assert low is None
    assert high is None


# ==========================================================
# Salary Anomaly
# ==========================================================

def test_salary_anomaly_missing():
    assert salary_anomaly_score("") == 0.0


def test_salary_anomaly_realistic():
    assert salary_anomaly_score(
        "60000-90000"
    ) == 1.0


def test_salary_anomaly_wide_range():
    score = salary_anomaly_score(
        "10000-200000"
    )

    assert score < 1.0


# ==========================================================
# Vagueness Score
# ==========================================================

def test_vagueness_empty():
    assert vagueness_score("") == 0.5


def test_vagueness_clean():
    assert (
        vagueness_score(
            "We are looking for an experienced software engineer."
        )
        == 0.0
    )


def test_vagueness_suspicious():
    text = (
        "Uncapped earnings! "
        "Be your own boss! "
        "No experience needed! "
        "Guaranteed income!"
    )

    score = vagueness_score(text)

    assert score > 0.1


# ==========================================================
# Missing Field Ratio
# ==========================================================

def test_missing_field_ratio_full():

    row = pd.Series(
        {
            "company_profile": "Acme Corp",
            "description": "Build stuff",
            "requirements": "Python",
            "benefits": "Health",
            "salary_range": "80000-100000",
            "industry": "Tech",
            "function": "Engineering",
            "required_experience": "Mid-level",
            "required_education": "Bachelor's",
        }
    )

    assert missing_field_ratio(row) == 0.0


def test_missing_field_ratio_empty():

    row = pd.Series(
        {
            k: ""
            for k in [
                "company_profile",
                "description",
                "requirements",
                "benefits",
                "salary_range",
                "industry",
                "function",
                "required_experience",
                "required_education",
            ]
        }
    )

    assert missing_field_ratio(row) == 1.0


# ==========================================================
# Requirement Compensation Mismatch
# ==========================================================

def test_req_comp_mismatch_no_mismatch():

    assert (
        req_comp_mismatch(
            "Python knowledge helpful",
            "80000-120000",
        )
        == 0.0
    )


def test_req_comp_mismatch_detected():

    requirements = (
        "PhD required. "
        "10+ years experience. "
        "Security clearance required. "
        "Master's degree."
    )

    assert (
        req_comp_mismatch(
            requirements,
            "",
        )
        == 1.0
    )


# ==========================================================
# Structured Feature Engineer
# ==========================================================

def test_structured_engineer_output_shape():

    df = pd.DataFrame(
        [
            {
                "title": "Data Scientist",
                "description": "Work with data",
                "requirements": "Python",
                "benefits": "Health insurance",
                "salary_range": "70000-100000",
                "company_profile": "Big Corp",
                "industry": "Tech",
                "function": "Data",
                "required_experience": "Mid level",
                "required_education": "Bachelor's",
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 0,
                "employment_type": "full-time",
            }
        ]
    )

    engineer = StructuredFeatureEngineer()

    result = engineer.fit_transform(df)

    assert result.shape == (1, 4)


def test_structured_engineer_numeric_output():

    df = pd.DataFrame(
        [
            {
                "title": "Developer",
                "description": "Build software",
                "requirements": "Python",
                "benefits": "Insurance",
                "salary_range": "60000-80000",
                "company_profile": "Company",
                "industry": "IT",
                "function": "Engineering",
                "required_experience": "Mid",
                "required_education": "Bachelor's",
                "telecommuting": 0,
                "has_company_logo": 1,
                "has_questions": 1,
                "employment_type": "full-time",
            }
        ]
    )

    engineer = StructuredFeatureEngineer()

    result = engineer.fit_transform(df)

    assert isinstance(result, np.ndarray)

    assert result.dtype == np.float32