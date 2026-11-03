
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf

from feature_extraction import extract_mfcc


# Load the saved model and training statistics.
# AI-assisted coding (ChatGPT, OpenAI):
# ChatGPT helped improve project path handling so the demo
# can use the original project or the repository folder.

local_project = Path.home() / "Desktop" / "Assessment001"
repository_project = Path(__file__).resolve().parent.parent


if (
    (repository_project / "models/final_speech_model.keras").is_file()
    and
    (repository_project / "models/normalisation_stats.npz").is_file()
):
    project = repository_project
else:
    project = local_project


model_file = project / "models/final_speech_model.keras"
stats_file = project / "models/normalisation_stats.npz"

if not model_file.is_file():
    raise FileNotFoundError(
        f"Trained CNN model not found: {model_file}"
    )

if not stats_file.is_file():
    raise FileNotFoundError(
        f"Normalisation statistics not found: {stats_file}"
    )

model = tf.keras.models.load_model(model_file)
print(f"Model loaded from: {model_file}")

stats = np.load(stats_file)
mean = stats["feature_mean"]
std = stats["feature_std"]
names = stats["names"]

sample_rate = 16000

demo_file = project / "data/demo/live_demo.wav"
demo_file.parent.mkdir(parents=True, exist_ok=True)


# 1. Predict a spoken name from a WAV file.
def predict_wav(wav_file, show_top3=False):
    """Predict a 2-second, 16 kHz WAV file using the saved CNN."""

    audio, sr = sf.read(wav_file)

    if sr != sample_rate or len(audio) != 2 * sample_rate:
        raise ValueError(
            "Expected a 2-second WAV at 16000 Hz."
        )

    # Extract MFCC features using the original implementation.
    mfcc = extract_mfcc(wav_file)

    if mfcc.shape != (199, 13):
        raise ValueError(
            f"Unexpected MFCC shape: {mfcc.shape}"
        )

    # Apply the saved training normalisation statistics.
    X = ((mfcc[None, ...] - mean) / std)[..., None]

    # ChatGPT helped explain CNN prediction scores
    # and why the highest score is not always correct.
    scores = model.predict(X, verbose=0)[0]
    best = int(np.argmax(scores))

    print(
        f"{Path(wav_file).name}: "
        f"{names[best]} ({scores[best] * 100:.2f}%)"
    )

    # Display the three highest-scoring predictions.
    if show_top3:
        print("Top 3:")

        for rank, i in enumerate(
            np.argsort(scores)[-3:][::-1], 1
        ):
            print(
                f"  {rank}. {names[i]}: "
                f"{scores[i] * 100:.2f}%"
            )


# 2. Choose the recording microphone.
# AI-assisted coding (ChatGPT, OpenAI):
# ChatGPT helped improve microphone device selection
# and input validation to prevent invalid device IDs.

def choose_microphone():
    print(sd.query_devices())

    while True:
        choice = input(
            "Microphone ID (Enter for default): "
        ).strip()

        if not choice:
            return None

        try:
            device = int(choice)
            sd.query_devices(device, "input")
            return device

        except (ValueError, sd.PortAudioError):
            print(
                "Invalid microphone device. "
                "Please select another ID."
            )


# 3. Record speech using the microphone.
def microphone_demo():
    """Record a spoken name and predict it."""

    device = choose_microphone()

    while True:
        choice = input(
            "\nEnter to record, q to quit: "
        ).strip().lower()

        if choice == "q":
            break

        # Start a three-second countdown.
        for second in (3, 2, 1):
            print(second)
            time.sleep(1)

        print("SPEAK NOW!")

        # Record two seconds of mono audio at 16 kHz.
        audio = sd.rec(
            32000,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device
        )

        sd.wait()

        # Calculate peak and RMS amplitudes.
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        print(f"Peak: {peak:.4f}, RMS: {rms:.4f}")

        # Reject recordings with very low volume.
        if peak < 0.01 or rms < 0.001:
            print("Audio is too quiet. Try again.")
            continue

        # Save the recording before extracting features.
        sf.write(
            demo_file,
            audio,
            sample_rate,
            subtype="PCM_16"
        )

        predict_wav(demo_file, show_top3=True)


# 4. Predict names from up to 10 WAV files.
def batch_demo():
    """Select between 1 and 10 WAV files for prediction."""

    # Open a file selection window.
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()

        files = filedialog.askopenfilenames(
            title="Select up to 10 WAV files",
            filetypes=[("WAV audio", "*.wav")]
        )

        root.destroy()

    except Exception:
        # Use manual file paths if the file picker is unavailable.
        print(
            "File picker unavailable. "
            "Enter WAV paths one at a time."
        )

        files = []

        for i in range(10):
            path = input(
                f"WAV {i + 1} path (Enter to finish): "
            ).strip().strip('"').strip("'")

            if not path:
                break

            files.append(
                str(Path(path).expanduser())
            )

    if not files:
        print("No files selected.")
        return

    if len(files) > 10:
        print(
            "Please select no more than 10 WAV files."
        )
        return

    # ChatGPT helped explain batch predictions
    # and how invalid WAV inputs are handled.
    print(
        f"\nBATCH PREDICTIONS ({len(files)} files)"
    )

    for wav_file in files:
        try:
            predict_wav(wav_file)

        except (OSError, ValueError) as error:
            print(
                f"{Path(wav_file).name}: "
                f"ERROR - {error}"
            )


# 5. Main demonstration menu.
if __name__ == "__main__":

    print("SPEECH RECOGNITION DEMO")
    print("Classes:", len(names))

    while True:
        mode = input(
            "\n1 = Microphone, "
            "2 = WAV files (1-10), "
            "q = Quit: "
        ).strip().lower()

        if mode == "1":
            microphone_demo()

        elif mode == "2":
            batch_demo()

        elif mode == "q":
            break

        else:
            print("Please choose 1, 2 or q.")
