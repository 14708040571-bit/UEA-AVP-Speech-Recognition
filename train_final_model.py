
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# 1. File paths
project = Path.home() / "Desktop" / "Assessment001"
dataset_file = project / "data/processed/train_mfcc.npz"

model_folder = project / "models"
model_folder.mkdir(parents=True, exist_ok=True)

model_file = model_folder / "final_speech_model.keras"
stats_file = model_folder / "normalisation_stats.npz"


# 2. Load all 400 training recordings
data = np.load(dataset_file)

X = data["X"].copy()
y = data["y"]
names = data["names"]

print("Training data:", X.shape)
print("Classes:", len(names))

if X.shape != (400, 199, 13) or len(names) != 20:
    raise ValueError("Unexpected training data")


# 3. Normalise features
mean = np.mean(X, axis=(0, 1), keepdims=True)
std = np.std(X, axis=(0, 1), keepdims=True)
std = np.where(std < 1e-8, 1.0, std)

X = ((X - mean) / std)[..., np.newaxis]

np.savez(
    stats_file,
    feature_mean=mean,
    feature_std=std,
    names=names
)


# 4. Build CNN
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

model = Sequential([
    Input(shape=X.shape[1:]),
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


# 5. Train final model for 16 epochs
model.fit(
    X, y,
    epochs=16,
    batch_size=16,
    verbose=1
)


# 6. Check training accuracy
train_loss, train_acc = model.evaluate(X, y, verbose=0)

print("\nFINAL TRAINING SUMMARY")
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Training loss: {train_loss:.4f}")
print("Epochs: 16")
print("Feature: MFCC")
print("CNN filters: 16, 32")
print("Learning rate: 0.001")
print("Dropout: 0.3")
print("Batch size: 16")


# 7. Save final model
model.save(model_file)

print("\nModel saved to:", model_file)
print("Normalisation stats saved to:", stats_file)
