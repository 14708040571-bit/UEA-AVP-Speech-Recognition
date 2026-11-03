
from pathlib import Path
import numpy as np
from feature_extraction import extract_mfcc


# Project folders.
project_folder = Path.home() / "Desktop" / "Assessment001"
train_folder = project_folder / "data" / "train"
processed_folder = project_folder / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)
output_file = processed_folder / "train_mfcc.npz"


# Find all training classes.
if not train_folder.is_dir():
    raise FileNotFoundError(
        f"Training folder not found: {train_folder}"
    )

class_names = sorted(
    folder.name
    for folder in train_folder.iterdir()
    if folder.is_dir()
)

print("\nDataset Preparation")
print("Training folder:", train_folder)
print("Number of classes:", len(class_names))
print("Class names:", class_names)


# Prepare containers.
features = []
labels = []
file_names = []


# ChatGPT helped simplify MFCC extraction and label collection.
print("\nExtracting MFCC features")

for class_index, class_name in enumerate(class_names):

    class_folder = train_folder / class_name
    wav_files = sorted(class_folder.glob("*.wav"))

    print(f"{class_name}: {len(wav_files)} files")

    for wav_file in wav_files:

        # Extract MFCC features using the existing implementation.
        mfcc = extract_mfcc(wav_file)

        features.append(mfcc)
        labels.append(class_index)
        file_names.append(wav_file.name)


# Convert the collected data into NumPy arrays.
X = np.array(features)
y = np.array(labels, dtype=np.int64)
names = np.array(class_names)
file_names = np.array(file_names)


# Display dataset information.
print("\nDataset shapes")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("names shape:", names.shape)
print("file_names shape:", file_names.shape)

print("\nTotal recordings:", len(X))
print("Number of classes:", len(names))


# Display the number of recordings per class.
print("\nRecordings per class")

for class_index, class_name in enumerate(names):
    count = np.sum(y == class_index)
    print(f"{class_name}: {count}")


# ChatGPT helped improve dataset validation checks.
if len(class_names) != 20:
    print(
        f"WARNING: Expected 20 classes, "
        f"but found {len(class_names)}."
    )

if len(X) != 400:
    print(
        f"WARNING: Expected 400 recordings, "
        f"but found {len(X)}."
    )

if X.ndim != 3:
    raise ValueError(
        "X does not have the expected "
        "(recordings, frames, coefficients) shape."
    )

if X.shape[1:] != (199, 13):
    print(
        "WARNING: Expected MFCC shape (199, 13), "
        f"but got {X.shape[1:]}."
    )


# Save the processed training dataset.
np.savez(
    output_file,
    X=X,
    y=y,
    names=names,
    file_names=file_names
)


# Final summary.
print("\nDataset saved successfully!")
print("Saved to:", output_file)
print("Final X shape:", X.shape)
print("Final y shape:", y.shape)
print("Number of classes:", len(names))
print("Number of file names:", len(file_names))
