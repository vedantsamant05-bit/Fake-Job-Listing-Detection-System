import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer

from src.config import TEXT_COLUMNS, CATEGORICAL_COLUMNS
from src.feature_engineering import StructuredFeatureEngineer

# Binary / boolean passthrough columns
BINARY_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate all text columns into a single text string.
    """
    return df[TEXT_COLUMNS].apply(
        lambda row: " ".join(row.values.astype(str)),
        axis=1,
    )


class TextCombiner:
    """
    Helper class for combining text columns.
    """

    def fit_transform(self, df):
        return combine_text_columns(df)

    def transform(self, df):
        return combine_text_columns(df)


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Branches:
    1. TF-IDF on combined text
    2. OneHotEncoding on categorical features
    3. Binary feature passthrough
    4. Custom engineered features
    """

    # ---------------------------------------------------------
    # Text branch
    # ---------------------------------------------------------

    text_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=3,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            ),
        )
    ])

    # ---------------------------------------------------------
    # Categorical branch
    # ---------------------------------------------------------

    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="unknown",
            ),
        ),
        (
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    # ---------------------------------------------------------
    # Engineered features branch
    # ---------------------------------------------------------

    eng_pipeline = Pipeline([
        ("engineer", StructuredFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])

    # ---------------------------------------------------------
    # Text combiner
    # ---------------------------------------------------------

    text_combiner = FunctionTransformer(
        func=combine_text_columns,
        validate=False,
    )

    # ---------------------------------------------------------
    # Main ColumnTransformer
    # ---------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                Pipeline([
                    ("combine", text_combiner),
                    ("tfidf", text_pipeline.named_steps["tfidf"]),
                ]),
                list(TEXT_COLUMNS),
            ),
            (
                "cat",
                cat_pipeline,
                CATEGORICAL_COLUMNS,
            ),
            (
                "binary",
                "passthrough",
                BINARY_COLUMNS,
            ),
            (
                "engineered",
                eng_pipeline,
                list(TEXT_COLUMNS)
                + ["salary_range"]
                + CATEGORICAL_COLUMNS
                + BINARY_COLUMNS,
            ),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    return preprocessor