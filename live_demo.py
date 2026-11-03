
from pathlib import Path
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from feature_extraction import extract_mfcc

# Load the saved model and training statistics.
project = Path.home() / "Desktop" / "Assessment001"
model = tf.keras.models.load_model(project / "models/final_speech_model.keras")
stats = np.load(project / "models/normalisation_stats.npz")
mean, std, names = stats["feature_mean"], stats["feature_std"], stats["names"]

sample_rate = 16000
demo_file = project / "data/demo/live_demo.wav"
demo_file.parent.mkdir(parents=True, exist_ok=True)


# 1. Predict a name from one WAV file.
def predict_wav(wav_file, show_top3=False):
    """Predict one 2-second, 16 kHz WAV file with the saved CNN."""

    audio, sr = sf.read(wav_file)

    if sr != sample_rate or len(audio) != 2 * sample_rate:
        raise ValueError("Expected a 2-second WAV at 16000 Hz.")

    # Extract 199 x 13 MFCC features.
    mfcc = extract_mfcc(wav_file)

    if mfcc.shape != (199, 13):
        raise ValueError(f"Unexpected MFCC shape: {mfcc.shape}")

    # Use saved normalisation values.
    X = ((mfcc[None, ...] - mean) / std)[..., None]

    # Predict with the trained CNN.
    scores = model.predict(X, verbose=0)[0]
    best = int(np.argmax(scores))

    print(
        f"{Path(wav_file).name}: "
        f"{names[best]} ({scores[best] * 100:.2f}%)"
    )

    # Show the three highest predictions.
    if show_top3:
        print("Top 3:")

        for rank, i in enumerate(np.argsort(scores)[-3:][::-1], 1):
            print(
                f"  {rank}. {names[i]}: "
                f"{scores[i] * 100:.2f}%"
            )


# 2. Record speech using the microphone.
def microphone_demo():
    """Record a name and predict it."""

    print(sd.query_devices())

    choice = input("Microphone ID (Enter for default): ").strip()
    device = int(choice) if choice else None

    while True:
        if input("\nEnter to record, q to quit: ").strip().lower() == "q":
            break

        # Countdown before recording.
        for second in (3, 2, 1):
            print(second)
            time.sleep(1)

        print("SPEAK NOW!")

        # Record two seconds at 16 kHz.
        audio = sd.rec(
            32000,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device
        )

        sd.wait()

        # Check the recording volume.
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))

        print(f"Peak: {peak:.4f}, RMS: {rms:.4f}")

        if peak < 0.01 or rms < 0.001:
            print("Audio is too quiet. Try again.")
            continue

        # Save the recording and predict the name.
        sf.write(demo_file, audio, sample_rate, subtype="PCM_16")

        predict_wav(demo_file, show_top3=True)


# 3. Predict names from up to 10 WAV files.
def batch_demo():
    """Select 1-10 WAV files and predict each name."""

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
        # Use manual file paths if the window is unavailable.
        print("File picker unavailable. Enter WAV paths one at a time.")

        files = []

        for i in range(10):
            path = input(
                f"WAV {i + 1} path (Enter to finish): "
            ).strip().strip('"').strip("'")

            if not path:
                break

            files.append(str(Path(path).expanduser()))

    if not files:
        print("No files selected.")
        return

    if len(files) > 10:
        print("Please select no more than 10 WAV files.")
        return

    # Predict every selected audio file.
    print(f"\nBATCH PREDICTIONS ({len(files)} files)")

    for wav_file in files:
        try:
            predict_wav(wav_file)

        except (OSError, ValueError) as error:
            print(f"{Path(wav_file).name}: ERROR - {error}")


# 4. Main menu.
if __name__ == "__main__":

    print("SPEECH RECOGNITION DEMO")
    print("Classes:", len(names))

    while True:
        mode = input(
            "\n1 = Microphone, 2 = WAV files (1-10), q = Quit: "
        ).strip().lower()

        if mode == "1":
            microphone_demo()

        elif mode == "2":
            batch_demo()

        elif mode == "q":
            break

        else:
            print("Please choose 1, 2 or q.")
