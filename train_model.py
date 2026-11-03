from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load data
project = Path.home() / "Desktop" / "Assessment001"
data = np.load(project / "data/processed/train_mfcc.npz")

X = data["X"]
y = data["y"]
names = data["names"]
file_names = data["file_names"]

print("Dataset:", X.shape)
print("Classes:", len(names))


# 2. Split training and validation data
train_idx, val_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val = X[train_idx].copy(), X[val_idx].copy()
y_train, y_val = y[train_idx], y[val_idx]

print("Training:", len(y_train))
print("Validation:", len(y_val))


# 3. Normalisation
mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)
std = np.where(std < 1e-8, 1.0, std)

X_train = ((X_train - mean) / std)[..., np.newaxis]
X_val = ((X_val - mean) / std)[..., np.newaxis]


# 4. Build CNN
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 5. Train CNN
stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[stop],
    verbose=1
)


# 6. Validation results
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
predictions = np.argmax(model.predict(X_val, verbose=0), axis=1)
best_epoch = np.argmin(history.history["val_loss"]) + 1

correct = np.sum(predictions == y_val)
wrong = np.sum(predictions != y_val)

print("\nVALIDATION RESULTS")
print(f"Accuracy: {val_acc * 100:.2f}%")
print(f"Loss: {val_loss:.4f}")
print(f"Best epoch: {best_epoch}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")


# 7. Show prediction errors
print("\nMisclassified samples:")

for i in range(len(y_val)):
    if predictions[i] != y_val[i]:
        print(
            file_names[val_idx[i]],
            names[y_val[i]], "->", names[predictions[i]]
        )


# 8. Confusion matrix
cm = confusion_matrix(y_val, predictions)

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names
).plot(ax=ax, xticks_rotation=90, values_format="d")

plt.title("Validation Confusion Matrix - MFCC CNN")
plt.tight_layout()
plt.show()


# 9. Accuracy and loss graphs
for metric in ["accuracy", "loss"]:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[metric], label="Training")
    plt.plot(history.history["val_" + metric], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title("Training and Validation " + metric.capitalize())
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


print("\nSELECTED MODEL")
print("Feature: MFCC")
print("CNN filters: 16, 32")
print("Learning rate: 0.001")
print("Dropout: 0.3")
print("Batch size: 16")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Best epoch: {best_epoch}")
