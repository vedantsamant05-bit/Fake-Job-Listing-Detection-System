import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train
from src.evaluate import full_evaluation, shap_summary

if __name__ == "__main__":

    pipeline, threshold, X_test, y_test = train(save=True)

    print("\n=== Running Evaluation ===")
    full_evaluation(
        pipeline,
        threshold,
        X_test,
        y_test,
    )

    print("\n=== Computing SHAP Summary ===")
    shap_summary(
        pipeline,
        X_test,
        max_samples=200,
    )

    print("\nTraining Complete.")