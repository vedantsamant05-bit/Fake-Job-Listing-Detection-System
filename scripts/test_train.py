from src.train import train

pipeline, threshold, X_test, y_test = train(save=False)

print("\nTraining completed.")
print("Threshold:", threshold)
print("Test samples:", len(X_test))