
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


# Project folders.
project_folder = Path.home() / "Desktop" / "Assessment001"
test_folder = project_folder / "data" / "test"

test_folder.mkdir(parents=True, exist_ok=True)


# Recording settings.
sample_rate = 16000
duration = 2.0
channels = 1
recordings_per_name = 5


# The same 20 names used in the training dataset.
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


# Display available audio devices.
print("\n========================================")
print("AVAILABLE AUDIO DEVICES")
print("========================================")

print(sd.query_devices())
print("\nCurrent default devices:", sd.default.device)


# ChatGPT helped improve microphone device input validation.
def choose_microphone():
    while True:
        device_input = input(
            "Microphone device ID (Enter for default): "
        ).strip()

        if not device_input:
            return None

        try:
            device_id = int(device_input)
        except ValueError:
            print("Please enter a valid numeric device ID.")
            continue

        try:
            sd.query_devices(device_id, "input")
            return device_id
        except (ValueError, sd.PortAudioError):
            print(
                "This device is not available for audio input. "
                "Please choose another device."
            )


input_device = choose_microphone()

print("\nSelected input device:", input_device)


# Record one independent test sample.
def record_one_file(output_file, name, recording_number):

    while True:

        print("\n----------------------------------------")
        print(f"Name: {name}")
        print(
            f"Recording: "
            f"{recording_number}/{recordings_per_name}"
        )
        print("----------------------------------------")

        input("Press Enter when you are ready...")

        print("\nRecording starts in:")

        for number in [3, 2, 1]:
            print(number)
            time.sleep(1)

        print("SPEAK NOW!")

        # Capture two seconds of mono audio at 16 kHz.
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=input_device
        )

        sd.wait()

        print("Recording finished.")

        # Calculate basic audio quality measurements.
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        print(f"Peak amplitude: {peak:.4f}")
        print(f"RMS amplitude: {rms:.4f}")

        # ChatGPT helped add recording confirmation and retry options.
        choice = input(
            "Save this recording? "
            "(y = save / r = retry / q = quit): "
        ).strip().lower()

        if choice == "y":

            # Save the accepted recording as a 16-bit WAV file.
            sf.write(
                output_file,
                audio,
                sample_rate,
                subtype="PCM_16"
            )

            print(f"Saved: {output_file}")
            return

        elif choice == "r":
            print("Recording discarded. Please try again.")

        elif choice == "q":
            print("Recording process stopped.")
            raise SystemExit

        else:
            print("Invalid option. Please enter y, r, or q.")


# Record the independent test dataset.
print("\n========================================")
print("INDEPENDENT TEST SET RECORDING")
print("========================================")

print("20 names x 5 recordings = 100 test recordings")
print("These recordings must NOT be added to training data.")


for name in names:

    # Create a separate test folder for each name.
    name_folder = test_folder / name
    name_folder.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print(f"NEXT NAME: {name}")
    print("========================================")

    for recording_number in range(
        1, recordings_per_name + 1
    ):

        output_file = (
            name_folder
            / f"{name}_test_{recording_number:02d}.wav"
        )

        # Avoid overwriting existing recordings without confirmation.
        if output_file.exists():

            print(f"\nFile already exists: {output_file}")

            choice = input(
                "Overwrite it? (y/n): "
            ).strip().lower()

            if choice != "y":
                print("Skipping this file.")
                continue

        record_one_file(
            output_file,
            name,
            recording_number
        )

    print(f"\nFinished test recordings for {name}.")


# Check the completed independent test dataset.
all_test_files = sorted(test_folder.rglob("*.wav"))

print("\n========================================")
print("TEST RECORDING COMPLETE")
print("========================================")

print("Total WAV files:", len(all_test_files))


# ChatGPT helped improve the recording count verification.
all_classes_complete = True

for name in names:

    name_folder = test_folder / name

    count = len(list(name_folder.glob("*.wav")))

    print(f"{name}: {count}/{recordings_per_name}")

    if count != recordings_per_name:
        all_classes_complete = False


if (
    len(all_test_files) == 100
    and all_classes_complete
):
    print(
        "\nSUCCESS: All 100 independent test recordings "
        "are present."
    )

else:
    print(
        "\nWARNING: The independent test dataset "
        "is incomplete or contains extra recordings."
    )

    print(
        "Expected: 20 names with exactly "
        "5 WAV files per name."
    )
