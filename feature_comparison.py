
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from feature_extraction import extract_mfcc


# 1. Settings
project = Path.home() / "Desktop" / "Assessment001"
train_folder = project / "data" / "train"

sample_rate = 16000
frame_length = 400
frame_step = 160
nfft = 512
n_filters = 26


# 2. Make a power spectrum for each short frame.
def power_spectrum(signal):

    length = len(signal)

    n_frames = max(
        1,
        1 + int(np.ceil((length - frame_length) / frame_step))
    )

    padded_length = (n_frames - 1) * frame_step + frame_length
    padded = np.pad(signal, (0, padded_length - length))

    indices = (
        np.arange(n_frames)[:, None] * frame_step
        + np.arange(frame_length)
    )

    frames = padded[indices] * np.hamming(frame_length)

    fft_result = np.fft.rfft(frames, n=nfft, axis=1)

    return np.abs(fft_result) ** 2 / nfft


# 3. Build rectangular filters at linear frequencies.
def linear_filterbank():

    bank = np.zeros((n_filters, nfft // 2 + 1))

    edges = np.linspace(
        0, nfft // 2 + 1, n_filters + 1, dtype=int
    )

    for i in range(n_filters):
        bank[i, edges[i]:edges[i + 1]] = 1.0

    return bank


# 4. Build overlapping triangular filters on the Mel scale.
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank():

    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(sample_rate / 2),
        n_filters + 2
    )

    hz_points = mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    bank = np.zeros((n_filters, nfft // 2 + 1))

    for m in range(1, n_filters + 1):
        left, centre, right = bins[m - 1:m + 2]

        # Rising and falling sides of the triangular filter.
        if centre > left:
            bank[m - 1, left:centre] = (
                np.arange(left, centre) - left
            ) / (centre - left)

        if right > centre:
            bank[m - 1, centre:right] = (
                right - np.arange(centre, right)
            ) / (right - centre)

    return bank


# 5. Turn band energies into log filterbank features.
def extract_filters(signal, bank):

    power = power_spectrum(signal)
    energy = np.dot(power, bank.T)
    energy = np.maximum(energy, np.finfo(float).eps)

    return np.log(energy)


linear_bank = linear_filterbank()
mel_bank = mel_filterbank()

class_names = sorted(
    folder.name for folder in train_folder.iterdir()
    if folder.is_dir()
)

linear_features = []
mel_features = []
mfcc_features = []
labels = []

print("Extracting features...")

for class_index, class_name in enumerate(class_names):

    wav_files = sorted(
        (train_folder / class_name).glob("*.wav")
    )

    print(class_name, len(wav_files), "files")

    for wav_file in wav_files:

        signal, sr = sf.read(wav_file)

        if sr != sample_rate:
            raise ValueError(f"Wrong sample rate: {wav_file}")

        if signal.ndim > 1:
            signal = signal[:, 0]

        linear_features.append(
            extract_filters(signal, linear_bank)
        )

        mel_features.append(
            extract_filters(signal, mel_bank)
        )

        mfcc_features.append(
            extract_mfcc(wav_file)
        )

        labels.append(class_index)


X_linear = np.array(linear_features)
X_mel = np.array(mel_features)
X_mfcc = np.array(mfcc_features)
y = np.array(labels)

print("Linear:", X_linear.shape)
print("Mel:", X_mel.shape)
print("MFCC:", X_mfcc.shape)


# AI-assisted learning (ChatGPT, OpenAI):
# ChatGPT helped explain fair validation across Linear, Mel and MFCC.
# 6. Use the same train/validation split for all features.
train_idx, val_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 7. Standardise the features and train the CNN.
def train_model(X, name):

    X_train = X[train_idx].copy()
    X_val = X[val_idx].copy()

    y_train = y[train_idx]
    y_val = y[val_idx]

    # Calculate statistics from training data only.
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = ((X_train - mean) / std)[..., np.newaxis]
    X_val = ((X_val - mean) / std)[..., np.newaxis]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    # Build a CNN with two convolutional layers.
    model = Sequential([
        Input(shape=X_train.shape[1:]),

        Conv2D(16, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(len(class_names), activation="softmax")
    ])

    # Set the optimiser and loss function.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Restore weights from the lowest validation loss.
    stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Train the CNN.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[stop],
        verbose=0
    )

    # Evaluate the validation accuracy.
    loss, accuracy = model.evaluate(
        X_val, y_val, verbose=0
    )

    best_epoch = np.argmin(
        history.history["val_loss"]
    ) + 1

    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Loss: {loss:.4f}")
    print(f"Best epoch: {best_epoch}")

    return {
        "name": name,
        "accuracy": accuracy,
        "loss": loss,
        "best_epoch": best_epoch
    }


# 8. Compare the three validation accuracies.
results = [
    train_model(X_linear, "Linear Rectangular"),
    train_model(X_mel, "Mel Triangular"),
    train_model(X_mfcc, "MFCC")
]

best = max(
    results,
    key=lambda result: result["accuracy"]
)

print("\nFINAL FEATURE COMPARISON")

for result in results:
    print(
        f"{result['name']:<20} "
        f"{result['accuracy'] * 100:.2f}% "
        f"| epoch {result['best_epoch']}"
    )

print("\nBest feature:", best["name"])

print(
    f"Best validation accuracy: "
    f"{best['accuracy'] * 100:.2f}%"
)
