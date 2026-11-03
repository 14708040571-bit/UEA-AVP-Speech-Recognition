from pathlib import Path

import numpy as np

from feature_extraction import extract_mfcc


# ============================================================
# 1. Project folders
# ============================================================

project_folder = (
    Path.home()
    / "Desktop"
    / "Assessment001"
)

test_folder = (
    project_folder
    / "data"
    / "test"
)

processed_folder = (
    project_folder
    / "data"
    / "processed"
)

processed_folder.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    processed_folder
    / "test_mfcc.npz"
)


# ============================================================
# 2. Load class names from final model normalisation file
# ============================================================

normalisation_file = (
    project_folder
    / "models"
    / "normalisation_stats.npz"
)

normalisation_data = np.load(
    normalisation_file
)

names = normalisation_data[
    "names"
]


print()
print("========================================")
print("PREPARE INDEPENDENT TEST DATASET")
print("========================================")

print(
    "Test folder:"
)

print(
    test_folder
)

print()

print(
    "Number of classes:",
    len(names)
)

print(
    "Class names:"
)

print(
    names
)


# ============================================================
# 3. Containers
# ============================================================

features = []

labels = []

file_names = []


# ============================================================
# 4. Extract MFCC from all test recordings
# ============================================================

print()
print("========================================")
print("EXTRACTING TEST MFCC FEATURES")
print("========================================")


for class_index, class_name in enumerate(
    names
):

    class_name = str(
        class_name
    )

    class_folder = (
        test_folder
        / class_name
    )


    if not class_folder.exists():

        raise FileNotFoundError(
            f"Test folder missing: {class_folder}"
        )


    wav_files = sorted(
        class_folder.glob(
            "*.wav"
        )
    )


    print()

    print(
        class_name,
        ":",
        len(wav_files),
        "files"
    )


    if len(wav_files) != 5:

        raise ValueError(
            f"{class_name} should have 5 test recordings, "
            f"but found {len(wav_files)}."
        )


    for wav_file in wav_files:

        mfcc = extract_mfcc(
            wav_file
        )


        if mfcc.shape != (
            199,
            13
        ):

            raise ValueError(
                f"Unexpected MFCC shape for "
                f"{wav_file.name}: {mfcc.shape}"
            )


        features.append(
            mfcc
        )

        labels.append(
            class_index
        )

        file_names.append(
            wav_file.name
        )


# ============================================================
# 5. Convert to NumPy arrays
# ============================================================

X_test = np.array(
    features
)

y_test = np.array(
    labels,
    dtype=np.int64
)

file_names = np.array(
    file_names
)


# ============================================================
# 6. Check final shapes
# ============================================================

print()
print("========================================")
print("TEST DATASET SHAPES")
print("========================================")

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "file_names shape:",
    file_names.shape
)

print(
    "names shape:",
    names.shape
)


if X_test.shape != (
    100,
    199,
    13
):

    raise ValueError(
        "Expected X_test shape "
        "(100, 199, 13), "
        f"but got {X_test.shape}"
    )


if y_test.shape != (
    100,
):

    raise ValueError(
        "Expected y_test shape "
        "(100,), "
        f"but got {y_test.shape}"
    )


# ============================================================
# 7. Save independent test dataset
# ============================================================

np.savez(
    output_file,
    X_test=X_test,
    y_test=y_test,
    names=names,
    file_names=file_names
)


print()
print("========================================")
print("TEST DATASET SAVED SUCCESSFULLY")
print("========================================")

print(
    "Saved to:"
)

print(
    output_file
)

print()

print(
    "Final X_test shape:",
    X_test.shape
)

print(
    "Final y_test shape:",
    y_test.shape
)

print(
    "Total test recordings:",
    len(X_test)
)

print()
print(
    "IMPORTANT:"
)

print(
    "This test dataset must not be used "
    "to retrain or tune the model."
)