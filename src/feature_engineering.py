import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Feature names (useful later for SHAP and debugging)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "salary_anomaly",
    "vagueness_score",
    "missing_field_ratio",
    "req_comp_mismatch",
]

# ---------------------------------------------------------------------------
# Salary anomaly
# ---------------------------------------------------------------------------

def parse_salary(salary_str: str):
    """
    Parse salary range strings such as:
    - 50000-80000
    - $50k-$80k
    - 60,000 - 90,000

    Returns:
        (low, high) as floats
        or (None, None) if parsing fails.
    """

    if not salary_str or str(salary_str).lower() in ("nan", "none"):
        return None, None

    salary_str = str(salary_str)

    salary_str = salary_str.replace(",", "").replace("$", "").lower()

    salary_str = re.sub(
        r"(\d+\.?\d*)k",
        lambda m: str(float(m.group(1)) * 1000),
        salary_str,
    )

    matches = re.findall(r"\d+(?:\.\d+)?", salary_str)

    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])

    return None, None


def salary_anomaly_score(salary_str: str) -> float:
    """
    Salary quality score.

    Returns:
        0.0 -> salary missing
        0.2 -> invalid salary
        0.3 -> absurdly wide range
        0.6 -> suspiciously wide range
        1.0 -> realistic salary
    """

    low, high = parse_salary(salary_str)

    if low is None or high is None:
        return 0.0

    if low <= 0 or high <= 0 or high < low:
        return 0.2

    ratio = high / (low + 1e-6)

    if ratio > 10:
        return 0.3

    if ratio > 5:
        return 0.6

    return 1.0


# ---------------------------------------------------------------------------
# Vagueness score
# ---------------------------------------------------------------------------

VAGUE_PHRASES = [
    r"\buncapped earnings\b",
    r"\bwork from anywhere\b",
    r"\bbe your own boss\b",
    r"\bno experience (needed|required|necessary)\b",
    r"\blimited time\b",
    r"\bguaranteed\b",
    r"\beasy money\b",
    r"\bpassive income\b",
    r"\bwork at home\b",
    r"\bmake \$\d+",
    r"\beach \w+ can earn\b",
    r"\bjoin our team today\b",
    r"\bopportunity of a lifetime\b",
    r"\bno degree required\b",
    r"\bfull training provided\b",
    r"\bgreat earning potential\b",
    r"\bmust be motivated\b",
    r"\bself-?starter\b",
    r"\bpassion for success\b",
]

VAGUE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in VAGUE_PHRASES]


def vagueness_score(text: str) -> float:
    """
    Measures how many suspicious marketing phrases appear.
    """

    if not text or not text.strip():
        return 0.5

    hits = sum(
        1 for pattern in VAGUE_PATTERNS
        if pattern.search(text)
    )

    return min(hits / len(VAGUE_PATTERNS), 1.0)


# ---------------------------------------------------------------------------
# Missing field ratio
# ---------------------------------------------------------------------------

IMPORTANT_FIELDS = [
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


def missing_field_ratio(row: pd.Series) -> float:
    """
    Fraction of important fields that are empty.
    """

    empty = sum(
        1
        for field in IMPORTANT_FIELDS
        if str(row.get(field, "")).strip().lower()
        in ("", "unknown", "nan")
    )

    return empty / len(IMPORTANT_FIELDS)


# ---------------------------------------------------------------------------
# Requirement–compensation mismatch
# ---------------------------------------------------------------------------

HIGH_SKILL_KEYWORDS = [
    r"\bphd\b",
    r"\bmaster'?s?\b",
    r"\badvanced degree\b",
    r"\b10\+?\s+years?\b",
    r"\bsenior\b",
    r"\blead\b",
    r"\barchitect\b",
    r"\bprincipal engineer\b",  # corrected spelling
    r"\bsecurity clearance\b",
    r"\bcertified\b",
]

HIGH_SKILL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in HIGH_SKILL_KEYWORDS
]


def req_comp_mismatch(
    requirements: str,
    salary_str: str,
) -> float:
    """
    High requirements + low/no salary = suspicious.
    """

    skill_level = sum(
        1
        for pattern in HIGH_SKILL_PATTERNS
        if pattern.search(requirements)
    )

    low, high = parse_salary(salary_str)

    has_salary = low is not None and high is not None

    if skill_level >= 2 and not has_salary:
        return 1.0

    if skill_level >= 3 and has_salary and high < 40000:
        return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# sklearn Transformer
# ---------------------------------------------------------------------------

class StructuredFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Output columns:

    [
        salary_anomaly,
        vagueness_score,
        missing_field_ratio,
        req_comp_mismatch
    ]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        results = []

        for _, row in X.iterrows():

            combined_text = " ".join([
                str(row.get("title", "")),
                str(row.get("description", "")),
                str(row.get("requirements", "")),
                str(row.get("benefits", "")),
            ])

            results.append([
                salary_anomaly_score(
                    str(row.get("salary_range", ""))
                ),
                vagueness_score(combined_text),
                missing_field_ratio(row),
                req_comp_mismatch(
                    str(row.get("requirements", "")),
                    str(row.get("salary_range", "")),
                ),
            ])

        return np.asarray(results, dtype=np.float32)

    def get_feature_names_out(self):
        return np.array(FEATURE_NAMES)