
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load the processed MFCC dataset.
project_folder = Path.home() / "Desktop" / "Assessment001"
data_file = project_folder / "data" / "processed" / "train_mfcc.npz"

data = np.load(data_file)

X = data["X"]
y = data["y"]
names = data["names"]

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)
print("Number of classes:", len(names))


# 2. Split the training and validation datasets.
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining X shape:", X_train.shape)
print("Training y shape:", y_train.shape)
print("Validation X shape:", X_val.shape)
print("Validation y shape:", y_val.shape)


# 3. Normalise MFCC features.
# AI-assisted coding (ChatGPT, OpenAI):
# ChatGPT helped simplify the normalisation code.
# The mean and standard deviation use training data only.

mfcc_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
mfcc_std = np.std(X_train, axis=(0, 1), keepdims=True)
mfcc_std = np.where(mfcc_std < 1e-8, 1.0, mfcc_std)

X_train = (X_train - mfcc_mean) / mfcc_std
X_val = (X_val - mfcc_mean) / mfcc_std

print("\nAfter normalisation:")
print("Training mean:", X_train.mean())
print("Training std:", X_train.std())
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# 4. Add the channel dimension for the 2D CNN.
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("\nAfter adding channel dimension:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# 5. Build the CNN for Experiment 1.
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

number_of_classes = len(names)

# ChatGPT helped simplify the CNN layer definitions.
# The original architecture and dropout rate are unchanged.
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(number_of_classes, activation="softmax")
])

print("\n==============================")
print("Experiment 1 CNN architecture")
print("Learning rate = 0.0005")
print("==============================")

model.summary()


# 6. Compile the model with a learning rate of 0.0005.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully!")
print("Learning rate: 0.0005")
print("Dropout: 0.3")
print("CNN filters: 16, 32")
print("Batch size: 16")


# 7. Train the model.
# ChatGPT assisted with understanding early stopping.
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping]
)

print("\nTraining finished!")


# 8. Evaluate the validation performance.
val_loss, val_accuracy = model.evaluate(
    X_val, y_val, verbose=0
)

best_epoch = np.argmin(history.history["val_loss"]) + 1

print("\n==============================")
print("Experiment 1 validation results")
print("Learning rate = 0.0005")
print("==============================")

print("Validation loss:", val_loss)
print("Validation accuracy:", val_accuracy)
print("Validation accuracy (%):", val_accuracy * 100)
print("Best epoch:", best_epoch)


# 9. Plot training and validation accuracy.
epochs_used = range(1, len(history.history["loss"]) + 1)

plt.figure()
plt.plot(
    epochs_used,
    history.history["accuracy"],
    label="Training accuracy"
)
plt.plot(
    epochs_used,
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Experiment 1 - Learning Rate 0.0005 - Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 10. Plot training and validation loss.
plt.figure()
plt.plot(
    epochs_used,
    history.history["loss"],
    label="Training loss"
)
plt.plot(
    epochs_used,
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Experiment 1 - Learning Rate 0.0005 - Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 11. Generate the validation confusion matrix.
prediction_probabilities = model.predict(X_val, verbose=0)
y_pred = np.argmax(prediction_probabilities, axis=1)

cm = confusion_matrix(y_val, y_pred)

fig, ax = plt.subplots(figsize=(12, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)

plt.title(
    "Experiment 1 - Learning Rate 0.0005 - Validation Confusion Matrix"
)
plt.tight_layout()
plt.show()


# 12. Display incorrectly classified validation samples.
# AI-assisted coding (ChatGPT, OpenAI):
# ChatGPT helped simplify the prediction error summary.

print("\n==============================")
print("Misclassified validation samples")
print("==============================")

wrong_count = 0

for i in range(len(y_val)):

    if y_pred[i] != y_val[i]:

        actual_name = names[y_val[i]]
        predicted_name = names[y_pred[i]]

        print(
            "Actual:", actual_name,
            "-> Predicted:", predicted_name
        )

        wrong_count += 1

correct_count = len(y_val) - wrong_count
calculated_accuracy = correct_count / len(y_val)

print("\nNumber of wrong predictions:", wrong_count)
print("Number of correct predictions:", correct_count)
print("Calculated validation accuracy:", calculated_accuracy)
print(
    "Calculated validation accuracy (%):",
    calculated_accuracy * 100
)


# 13. Print the experiment summary.
print("\n==============================")
print("Experiment 1 Summary")
print("==============================")

print("Learning rate: 0.0005")
print("Dropout: 0.3")
print("CNN filters: 16, 32")
print("Batch size: 16")
print("Validation loss:", val_loss)
print("Validation accuracy (%):", val_accuracy * 100)
print("Best epoch:", best_epoch)
