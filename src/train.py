import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.config import (
    PIPELINE_PATH,
    THRESHOLD_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    SMOTE_SAMPLING_STRATEGY,
    TARGET_COLUMN,
    THRESHOLD_MIN_PRECISION,
)

from src.data_loader import load_raw_data, basic_clean
from src.preprocessing import build_preprocessor


# ==========================================================
# Models
# ==========================================================

def get_classifiers():
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=20,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ==========================================================
# Threshold Tuning
# ==========================================================

def tune_threshold(
    y_true,
    y_proba,
    min_precision=THRESHOLD_MIN_PRECISION,
):
    """
    Maximise recall while maintaining a minimum precision.
    """

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        y_proba,
    )

    best_threshold = 0.5
    best_recall = 0.0

    for p, r, t in zip(
        precisions[:-1],
        recalls[:-1],
        thresholds,
    ):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_threshold = t

    return float(best_threshold)


# ==========================================================
# Training
# ==========================================================

def train(data_path=None, save=True):

    print("=== Loading data ===")

    df = load_raw_data(data_path)
    df = basic_clean(df)

    X = df.drop(
        columns=[TARGET_COLUMN, "job_id"],
        errors="ignore",
    )

    y = df[TARGET_COLUMN].astype(int)

    print("\nClass distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTrain size: {X_train.shape}")
    print(f"Test size : {X_test.shape}")

    preprocessor = build_preprocessor()

    classifiers = get_classifiers()

    results = {}

    for name, clf in classifiers.items():

        print(f"\n{'=' * 50}")
        print(f"Training: {name}")
        print(f"{'=' * 50}")

        pipeline = ImbPipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "smote",
                SMOTE(
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                    random_state=RANDOM_STATE,
                    k_neighbors=5,
                ),
            ),
            (
                "classifier",
                clf,
            ),
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        print(
            f"CV ROC-AUC: "
            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_proba,
        )

        threshold = tune_threshold(
            y_test,
            y_proba,
        )

        y_pred = (y_proba >= threshold).astype(int)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
        )

        fake_metrics = report.get("1", {})

        precision = fake_metrics.get("precision", 0)
        recall = fake_metrics.get("recall", 0)
        f1 = fake_metrics.get("f1-score", 0)

        print(f"Test ROC-AUC     : {roc_auc:.4f}")
        print(f"Best Threshold   : {threshold:.4f}")
        print(f"Fake Precision   : {precision:.4f}")
        print(f"Fake Recall      : {recall:.4f}")
        print(f"Fake F1 Score    : {f1:.4f}")

        results[name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "roc_auc": roc_auc,
            "recall_fake": recall,
            "f1_fake": f1,
        }

    # ======================================================
    # Best Model Selection
    # ======================================================

    best_name = max(
        results,
        key=lambda x: results[x]["recall_fake"],
    )

    best = results[best_name]

    print("\n" + "=" * 60)
    print(
        f"BEST MODEL: {best_name} "
        f"(Recall={best['recall_fake']:.4f})"
    )
    print("=" * 60)

    # ======================================================
    # Save Artifacts
    # ======================================================

    if save:

        joblib.dump(
            best["pipeline"],
            PIPELINE_PATH,
        )

        joblib.dump(
            best["threshold"],
            THRESHOLD_PATH,
        )

        metrics = {
            "model": best_name,
            "roc_auc": float(best["roc_auc"]),
            "recall_fake": float(best["recall_fake"]),
            "f1_fake": float(best["f1_fake"]),
            "threshold": float(best["threshold"]),
        }

        with open(
            METRICS_PATH,
            "w",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=4,
            )

        print(f"\nSaved pipeline  : {PIPELINE_PATH}")
        print(f"Saved threshold : {THRESHOLD_PATH}")
        print(f"Saved metrics   : {METRICS_PATH}")

    return (
        best["pipeline"],
        best["threshold"],
        X_test,
        y_test,
    )


if __name__ == "__main__":
    train()