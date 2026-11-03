from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


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

test_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Recording settings
# ============================================================

sample_rate = 16000

duration = 2.0

channels = 1

recordings_per_name = 5


# ============================================================
# 3. The same 20 names used in training
# ============================================================

names = [
    "Aaron",
    "Alex",
    "Ali",
    "Charlie",
    "Harriet",
    "Jacek",
    "Jack",
    "Jade",
    "Joshua",
    "Mason",
    "Nathan",
    "Nicholas",
    "Sam",
    "Samuel",
    "Sean",
    "Toby",
    "Tommy",
    "Xingchen",
    "Ziang",
    "Zihan"
]


# ============================================================
# 4. Show recording devices
# ============================================================

print()
print("========================================")
print("AVAILABLE AUDIO DEVICES")
print("========================================")
print()

print(
    sd.query_devices()
)

print()
print(
    "Current default devices:",
    sd.default.device
)


# ============================================================
# 5. Choose microphone
# ============================================================

print()
print(
    "Enter microphone device ID."
)

print(
    "If you want to use the current default input device,"
)

print(
    "just press Enter."
)


device_input = input(
    "Microphone device ID: "
).strip()


if device_input == "":

    input_device = None

else:

    input_device = int(
        device_input
    )


print()
print(
    "Selected input device:",
    input_device
)


# ============================================================
# 6. Recording function
# ============================================================

def record_one_file(
    output_file,
    name,
    recording_number
):

    print()
    print("----------------------------------------")

    print(
        f"Name: {name}"
    )

    print(
        f"Recording: "
        f"{recording_number}/{recordings_per_name}"
    )

    print("----------------------------------------")

    input(
        "Press Enter when you are ready..."
    )


    print()
    print("Recording starts in:")

    for number in [
        3,
        2,
        1
    ]:

        print(
            number
        )

        time.sleep(
            1
        )


    print("SPEAK NOW!")


    audio = sd.rec(

        int(
            duration
            * sample_rate
        ),

        samplerate=sample_rate,

        channels=channels,

        dtype="float32",

        device=input_device
    )


    sd.wait()


    print(
        "Recording finished."
    )


    # --------------------------------------------------------
    # Basic recording information
    # --------------------------------------------------------

    peak = np.max(
        np.abs(audio)
    )

    rms = np.sqrt(
        np.mean(
            audio ** 2
        )
    )


    print(
        f"Peak amplitude: {peak:.4f}"
    )

    print(
        f"RMS amplitude: {rms:.4f}"
    )


    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    sf.write(

        output_file,

        audio,

        sample_rate,

        subtype="PCM_16"
    )


    print(
        "Saved:"
    )

    print(
        output_file
    )


# ============================================================
# 7. Record the independent test set
# ============================================================

print()
print("========================================")
print("INDEPENDENT TEST SET RECORDING")
print("========================================")

print()
print(
    "20 names x 5 recordings = 100 test recordings"
)

print(
    "These recordings must NOT be added to training data."
)

print()


for name in names:

    # --------------------------------------------------------
    # Create one folder per name
    # --------------------------------------------------------

    name_folder = (
        test_folder
        / name
    )

    name_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print()
    print("========================================")

    print(
        f"NEXT NAME: {name}"
    )

    print("========================================")


    for recording_number in range(
        1,
        recordings_per_name + 1
    ):

        output_file = (
            name_folder
            / (
                f"{name}_test_"
                f"{recording_number:02d}.wav"
            )
        )


        # ----------------------------------------------------
        # Do not silently overwrite an existing test file
        # ----------------------------------------------------

        if output_file.exists():

            print()
            print(
                "File already exists:"
            )

            print(
                output_file
            )


            choice = input(
                "Overwrite it? (y/n): "
            ).strip().lower()


            if choice != "y":

                print(
                    "Skipping this file."
                )

                continue


        record_one_file(

            output_file,

            name,

            recording_number
        )


    print()
    print(
        f"Finished all test recordings for {name}."
    )


# ============================================================
# 8. Final check
# ============================================================

all_test_files = sorted(
    test_folder.rglob(
        "*.wav"
    )
)


print()
print()
print("========================================")
print("TEST RECORDING COMPLETE")
print("========================================")

print(
    "Total WAV files:",
    len(all_test_files)
)


for name in names:

    name_folder = (
        test_folder
        / name
    )

    count = len(
        list(
            name_folder.glob(
                "*.wav"
            )
        )
    )


    print(
        f"{name}: {count}"
    )


if len(all_test_files) == 100:

    print()
    print(
        "SUCCESS: 100 independent test recordings found."
    )

else:

    print()
    print(
        "WARNING: Expected 100 test recordings, "
        f"but found {len(all_test_files)}."
    )