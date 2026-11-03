
from pathlib import Path
import numpy as np
from feature_extraction import extract_mfcc


# Project folders.
project_folder = Path.home() / "Desktop" / "Assessment001"
test_folder = project_folder / "data" / "test"
processed_folder = project_folder / "data" / "processed"

processed_folder.mkdir(parents=True, exist_ok=True)
output_file = processed_folder / "test_mfcc.npz"


# Load class names from the saved model statistics.
normalisation_file = (
    project_folder / "models" / "normalisation_stats.npz"
)

normalisation_data = np.load(normalisation_file)
names = normalisation_data["names"]

print("\nPREPARE INDEPENDENT TEST DATASET")
print("Test folder:", test_folder)
print("Number of classes:", len(names))
print("Class names:", names)


# Prepare data containers.
features = []
labels = []
file_names = []


# ChatGPT helped simplify the test folder validation.
if not test_folder.is_dir():
    raise FileNotFoundError(
        f"Test folder not found: {test_folder}"
    )

print("\nEXTRACTING TEST MFCC FEATURES")

for class_index, class_name in enumerate(names):

    class_name = str(class_name)
    class_folder = test_folder / class_name

    if not class_folder.is_dir():
        raise FileNotFoundError(
            f"Test folder missing: {class_folder}"
        )

    wav_files = sorted(class_folder.glob("*.wav"))
    print(f"{class_name}: {len(wav_files)} files")

    if len(wav_files) != 5:
        raise ValueError(
            f"{class_name} should have 5 test recordings, "
            f"but found {len(wav_files)}."
        )

    for wav_file in wav_files:

        # Extract MFCC features using the original function.
        mfcc = extract_mfcc(wav_file)

        # ChatGPT helped simplify MFCC shape validation.
        if mfcc.shape != (199, 13):
            raise ValueError(
                f"Unexpected MFCC shape for "
                f"{wav_file.name}: {mfcc.shape}"
            )

        features.append(mfcc)
        labels.append(class_index)
        file_names.append(wav_file.name)


# Convert the collected data into NumPy arrays.
X_test = np.array(features)
y_test = np.array(labels, dtype=np.int64)
file_names = np.array(file_names)


# Display dataset shapes.
print("\nTEST DATASET SHAPES")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("file_names shape:", file_names.shape)
print("names shape:", names.shape)


# ChatGPT helped simplify the final dataset checks.
if X_test.shape != (100, 199, 13):
    raise ValueError(
        f"Expected X_test shape (100, 199, 13), "
        f"but got {X_test.shape}"
    )

if y_test.shape != (100,):
    raise ValueError(
        f"Expected y_test shape (100,), "
        f"but got {y_test.shape}"
    )


# Save the independent test dataset.
np.savez(
    output_file,
    X_test=X_test,
    y_test=y_test,
    names=names,
    file_names=file_names
)


# Final summary.
print("\nTEST DATASET SAVED SUCCESSFULLY")
print("Saved to:", output_file)
print("Final X_test shape:", X_test.shape)
print("Final y_test shape:", y_test.shape)
print("Total test recordings:", len(X_test))

print(
    "\nIMPORTANT: This test dataset must not be used "
    "to retrain or tune the model."
)
