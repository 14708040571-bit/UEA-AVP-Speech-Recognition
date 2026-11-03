from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# 1. Find processed MFCC dataset
# ==========================================

project_folder = (
    Path.home()
    / "Desktop"
    / "Assessment001"
)

data_file = (
    project_folder
    / "data"
    / "processed"
    / "train_mfcc.npz"
)


# ==========================================
# 2. Load X, y and class names
# ==========================================

data = np.load(data_file)

X = data["X"]
y = data["y"]
names = data["names"]


print("Original X shape:", X.shape)
print("Original y shape:", y.shape)
print("Number of classes:", len(names))


# ==========================================
# 3. Split into Training and Validation
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print()
print("Training X shape:", X_train.shape)
print("Training y shape:", y_train.shape)

print()
print("Validation X shape:", X_val.shape)
print("Validation y shape:", y_val.shape)


# ==========================================
# 4. Normalise MFCC features
# ==========================================

mfcc_mean = np.mean(
    X_train,
    axis=(0, 1),
    keepdims=True
)

mfcc_std = np.std(
    X_train,
    axis=(0, 1),
    keepdims=True
)

mfcc_std = np.where(
    mfcc_std < 1e-8,
    1.0,
    mfcc_std
)

X_train = (
    X_train - mfcc_mean
) / mfcc_std

X_val = (
    X_val - mfcc_mean
) / mfcc_std


print()
print("After normalisation:")
print("Training mean:", X_train.mean())
print("Training std:", X_train.std())
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# ==========================================
# 5. Add channel dimension for 2D CNN
# ==========================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]


print()
print("After adding channel dimension:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)


# ==========================================
# 6. Build Experiment 1 CNN
# Only learning rate is changed
# ==========================================

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(42)

number_of_classes = len(names)


model = Sequential([

    Input(
        shape=X_train.shape[1:]
    ),

    Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.3
    ),

    Dense(
        number_of_classes,
        activation="softmax"
    )
])


# ==========================================
# 7. Display neural network architecture
# ==========================================

print()
print("==============================")
print("Experiment 1 CNN architecture")
print("Learning rate = 0.0005")
print("==============================")

model.summary()


# ==========================================
# 8. Compile the model
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print()
print("Model compiled successfully!")
print("Learning rate: 0.0005")
print("Dropout: 0.3")
print("CNN filters: 16, 32")
print("Batch size: 16")


# ==========================================
# 9. Train the model
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=16,

    callbacks=[
        early_stopping
    ]
)


print()
print("Training finished!")


# ==========================================
# 10. Evaluate Experiment 1
# ==========================================

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)


print()
print("==============================")
print("Experiment 1 validation results")
print("Learning rate = 0.0005")
print("==============================")

print(
    "Validation loss:",
    val_loss
)

print(
    "Validation accuracy:",
    val_accuracy
)

print(
    "Validation accuracy (%):",
    val_accuracy * 100
)


best_epoch = (
    np.argmin(
        history.history["val_loss"]
    )
    + 1
)


print(
    "Best epoch:",
    best_epoch
)


# ==========================================
# 11. Plot training history
# ==========================================

epochs_used = range(
    1,
    len(history.history["loss"]) + 1
)


# ---------- Accuracy ----------

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

plt.title(
    "Experiment 1 - Learning Rate 0.0005 - Accuracy"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ---------- Loss ----------

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

plt.title(
    "Experiment 1 - Learning Rate 0.0005 - Loss"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ==========================================
# 12. Confusion Matrix
# ==========================================

prediction_probabilities = model.predict(
    X_val,
    verbose=0
)


y_pred = np.argmax(
    prediction_probabilities,
    axis=1
)


cm = confusion_matrix(
    y_val,
    y_pred
)


fig, ax = plt.subplots(
    figsize=(12, 12)
)


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


# ==========================================
# 13. Print misclassified validation samples
# ==========================================

print()
print("==============================")
print("Misclassified validation samples")
print("==============================")


wrong_count = 0


for i in range(
    len(y_val)
):

    if y_pred[i] != y_val[i]:

        actual_name = names[
            y_val[i]
        ]

        predicted_name = names[
            y_pred[i]
        ]

        print(
            "Actual:",
            actual_name,
            "-> Predicted:",
            predicted_name
        )

        wrong_count += 1


correct_count = (
    len(y_val)
    - wrong_count
)


print()
print(
    "Number of wrong predictions:",
    wrong_count
)

print(
    "Number of correct predictions:",
    correct_count
)


calculated_accuracy = (
    correct_count
    / len(y_val)
)


print(
    "Calculated validation accuracy:",
    calculated_accuracy
)

print(
    "Calculated validation accuracy (%):",
    calculated_accuracy * 100
)


# ==========================================
# 14. Experiment summary
# ==========================================

print()
print("==============================")
print("Experiment 1 Summary")
print("==============================")

print("Learning rate: 0.0005")
print("Dropout: 0.3")
print("CNN filters: 16, 32")
print("Batch size: 16")
print("Validation loss:", val_loss)
print("Validation accuracy (%):", val_accuracy * 100)
print("Best epoch:", best_epoch)