from src.train import train
from src.evaluate import full_evaluation

print("Loading test data and model...")

pipeline, threshold, X_test, y_test = train(save=False)

print("\nRunning evaluation...")

full_evaluation(
    pipeline,
    threshold,
    X_test,
    y_test,
)

print("\nEvaluation complete.")