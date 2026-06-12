import joblib
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from src.config import (
    PIPELINE_PATH,
    THRESHOLD_PATH,
    MODEL_DIR,
)


# ==========================================================
# Load Artifacts
# ==========================================================

def load_artifacts():
    """
    Load trained pipeline and threshold.
    """

    pipeline = joblib.load(PIPELINE_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return pipeline, threshold


# ==========================================================
# Full Evaluation
# ==========================================================

def full_evaluation(
    pipeline,
    threshold,
    X_test,
    y_test,
):
    """
    Generate evaluation metrics and plots.
    """

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Real", "Fake"],
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    print("\n=== Classification Report ===\n")
    print(report)

    print(f"\nAUROC: {roc_auc:.4f}")

    report_path = MODEL_DIR / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    print(
        f"\nSaved classification report to "
        f"{report_path}"
    )

    # ------------------------------------------------------
    # Evaluation Plots
    # ------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=axes[0],
        display_labels=["Real", "Fake"],
    )

    axes[0].set_title(
        "Confusion Matrix"
    )

    RocCurveDisplay.from_predictions(
        y_test,
        y_proba,
        ax=axes[1],
    )

    axes[1].set_title(
        "ROC Curve"
    )

    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_proba,
        ax=axes[2],
    )

    axes[2].set_title(
        "Precision-Recall Curve"
    )

    plt.tight_layout()

    out_path = (
        MODEL_DIR /
        "model_evaluation.png"
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved evaluation plots to "
        f"{out_path}"
    )

    plt.show()


# ==========================================================
# SHAP
# ==========================================================

def compute_shap_values(
    pipeline,
    X_sample,
    max_samples=200,
):
    """
    Compute SHAP values.
    """

    if shap is None:
        raise ImportError(
            "SHAP is not installed. "
            "Run: pip install shap"
        )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    clf = pipeline.named_steps[
        "classifier"
    ]

    X_transformed = (
        preprocessor.transform(
            X_sample.iloc[:max_samples]
        )
    )

    # Logistic Regression
    if isinstance(
        clf,
        LogisticRegression,
    ):

        explainer = shap.LinearExplainer(
            clf,
            X_transformed,
        )

        shap_values = (
            explainer.shap_values(
                X_transformed
            )
        )

    else:

        explainer = shap.TreeExplainer(
            clf
        )

        shap_values = (
            explainer.shap_values(
                X_transformed
            )
        )

        if isinstance(
            shap_values,
            list,
        ):
            shap_values = shap_values[1]

    return (
        shap_values,
        X_transformed,
    )


# ==========================================================
# SHAP Summary Plot
# ==========================================================

def shap_summary(
    pipeline,
    X_sample,
    feature_names=None,
    max_samples=200,
):

    if shap is None:

        print(
            "SHAP not installed. "
            "Skipping SHAP summary."
        )

        return

    (
        shap_values,
        X_transformed,
    ) = compute_shap_values(
        pipeline,
        X_sample,
        max_samples=max_samples,
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    out_path = (
        MODEL_DIR /
        "shap_summary.png"
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved SHAP summary to "
        f"{out_path}"
    )

    plt.show()