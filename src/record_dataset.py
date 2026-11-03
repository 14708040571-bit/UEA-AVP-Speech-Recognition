
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import time


# Define the 20 spoken-name classes.
names = [
    "Aaron",
    "Alex",
    "Ali",
    "Charlie",
    "Jacek",
    "Jack",
    "Jade",
    "Joshua",
    "Harriet",
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


print("Assessment 001")
print("Dataset setup started")
print("Number of names:", len(names))


# Set the project and training data paths.
project_folder = Path.home() / "Desktop" / "Assessment001"
train_folder = project_folder / "data" / "train"

print("Training folder:")
print(train_folder)


# Create a separate folder for each name.
for name in names:
    name_folder = train_folder / name
    name_folder.mkdir(parents=True, exist_ok=True)
    print("Created:", name_folder)

print("Dataset folders are ready!")


# Audio recording settings.
fs = 16000
duration = 2


# ChatGPT helped write this function to support audio device selection.
def choose_device(label):
    while True:
        answer = input(
            f"{label} device ID (Enter for default): "
        ).strip()

        if not answer:
            return None

        try:
            return int(answer)

        except ValueError:
            print(
                "Please enter a valid device ID "
                "or press Enter."
            )


print("\nAvailable audio devices:")
print(sd.query_devices())

mic_device = choose_device("Microphone")
playback_device = choose_device("Playback")


print("\nStarting training data collection.")
print("Target: 20 recordings per name.")
print("Total: 400 recordings.")


# Record the training samples for each name.
for test_name in names:

    name_folder = train_folder / test_name

    # Find the next available recording number.
    next_number = 1

    while (
        name_folder / f"{test_name}_{next_number:02d}.wav"
    ).exists():
        next_number += 1

    if next_number > 20:
        print(
            f"\n{test_name}: 20 recordings already exist. "
            "Skipping."
        )
        continue

    while next_number <= 20:

        output_file = (
            name_folder
            / f"{test_name}_{next_number:02d}.wav"
        )

        while True:

            print("\n--------------------------------")
            print(
                f"Preparing to record: "
                f"{test_name}_{next_number:02d}.wav"
            )
            print(f"Progress: {next_number}/20")
            print(f"Please say: {test_name}")
            print("--------------------------------")

            # Three-second countdown.
            for i in range(3, 0, -1):
                print(i)
                time.sleep(1)

            print("Recording...")

            recording = sd.rec(
                int(duration * fs),
                samplerate=fs,
                channels=1,
                device=mic_device
            )

            sd.wait()

            print("Recording finished!")

            print(
                "Maximum amplitude:",
                recording.max()
            )
            print(
                "Minimum amplitude:",
                recording.min()
            )

            # Play the recording for quality checking.
            print("Playing recording...")

            sd.play(
                recording,
                fs,
                device=playback_device
            )

            sd.wait()

            print("Playback finished!")

            choice = input(
                "Save this recording? "
                "(y = save / r = retry / q = quit): "
            ).strip().lower()

            if choice == "y":

                sf.write(
                    output_file,
                    recording,
                    fs
                )

                print(
                    f"Saved successfully: {output_file.name}"
                )

                next_number += 1
                break

            elif choice == "r":

                print(
                    "Recording discarded. "
                    "Please try again."
                )

            elif choice == "q":

                print("Recording process stopped.")
                raise SystemExit

            else:

                print(
                    "Invalid option. "
                    "Please enter y, r, or q."
                )

    print("\n==============================")
    print(f"{test_name}: 20 recordings completed.")
    print("==============================")


print("\nTraining data collection completed!")
