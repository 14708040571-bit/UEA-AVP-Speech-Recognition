
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 1. File paths
project = Path.home() / "Desktop" / "Assessment001"
results = project / "results"
results.mkdir(exist_ok=True)

test_file = project / "data/processed/test_mfcc.npz"
model_file = project / "models/final_speech_model.keras"
stats_file = project / "models/normalisation_stats.npz"

csv_file = results / "final_test_predictions.csv"
cm_file = results / "final_test_confusion_matrix.png"


# 2. Load test data and training statistics
data = np.load(test_file)
stats = np.load(stats_file)

X_test = data["X_test"].copy()
y_test = data["y_test"]
names = data["names"]
file_names = data["file_names"]

print("Test data:", X_test.shape)
print("Classes:", len(names))

if X_test.shape != (100, 199, 13) or y_test.shape != (100,):
    raise ValueError("Unexpected test data shape")

if not np.array_equal(names, stats["names"]):
    raise ValueError("Class names do not match")


# 3. Normalise test data
mean = stats["feature_mean"]
std = stats["feature_std"]

X_test = ((X_test - mean) / std)[..., np.newaxis]


# 4. Load model and evaluate
model = tf.keras.models.load_model(model_file)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)
confidence = np.max(probabilities, axis=1)

correct = np.sum(predictions == y_test)
wrong = np.sum(predictions != y_test)

print("\nINDEPENDENT TEST RESULTS")
print(f"Accuracy: {test_acc * 100:.2f}%")
print(f"Loss: {test_loss:.4f}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")


# 5. Accuracy for each name
print("\nPer-class accuracy:")

for i, name in enumerate(names):
    mask = y_test == i
    total = np.sum(mask)
    class_correct = np.sum(predictions[mask] == y_test[mask])
    accuracy = class_correct / total * 100

    print(f"{name}: {class_correct}/{total} = {accuracy:.0f}%")


# 6. Confusion matrix
cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(14, 14))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names
).plot(ax=ax, xticks_rotation=90, values_format="d")

plt.title("Independent Test Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_file, dpi=300, bbox_inches="tight")
plt.show()


# 7. Save predictions to CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "File", "Actual", "Predicted",
        "Confidence_Percent", "Correct"
    ])

    for i in range(len(y_test)):
        writer.writerow([
            str(file_names[i]),
            str(names[y_test[i]]),
            str(names[predictions[i]]),
            f"{confidence[i] * 100:.2f}",
            predictions[i] == y_test[i]
        ])


# 8. Final summary
print("\nFINAL SYSTEM SUMMARY")
print("Training recordings: 400")
print("Independent test recordings: 100")
print("Classes: 20")
print("Feature: 13 MFCC coefficients")
print("Validation accuracy: 81.25%")
print(f"Independent test accuracy: {test_acc * 100:.2f}%")
print("Confusion matrix:", cm_file)
print("Prediction CSV:", csv_file)
