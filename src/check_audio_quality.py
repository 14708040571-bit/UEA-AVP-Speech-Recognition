
from pathlib import Path
import csv
import numpy as np
import soundfile as sf


# Project paths.
project_folder = Path.home() / "Desktop" / "Assessment001"
train_folder = project_folder / "data" / "train"
results_folder = project_folder / "results"
results_folder.mkdir(parents=True, exist_ok=True)

report_file = results_folder / "audio_quality_report.csv"


# Expected recording settings.
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_DURATION = 2.0
DURATION_TOLERANCE = 0.05
QUIET_RMS_THRESHOLD = 0.005
CLIPPING_THRESHOLD = 0.999
MIN_CLIPPED_SAMPLES = 5
EXPECTED_FILES_PER_CLASS = 20


# Find training recordings.
if not train_folder.is_dir():
    raise FileNotFoundError(f"Training folder not found: {train_folder}")

wav_files = sorted(train_folder.glob("*/*.wav"))
class_folders = sorted(
    p for p in train_folder.iterdir() if p.is_dir()
)

print("\nAudio Quality Check")
print("Training folder:", train_folder)
print("Total WAV files found:", len(wav_files))


# Check recording counts for each class.
class_count_problems = []

print("\nFiles per class")

for folder in class_folders:
    count = len(list(folder.glob("*.wav")))
    print(f"{folder.name}: {count}")

    if count != EXPECTED_FILES_PER_CLASS:
        class_count_problems.append((folder.name, count))


# Initialise the quality report.
report_rows = []
flagged_files = []

read_error_count = 0
sample_rate_problem_count = 0
duration_problem_count = 0
quiet_count = 0
clipping_count = 0


# ChatGPT helped simplify the audio measurement calculations.
for wav_file in wav_files:
    class_name = wav_file.parent.name
    status_list = []

    try:
        audio, sample_rate = sf.read(
            wav_file, dtype="float32"
        )

        # Convert stereo audio to mono.
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        number_of_samples = len(audio)
        duration = number_of_samples / sample_rate

        peak = (
            float(np.max(np.abs(audio)))
            if number_of_samples > 0 else 0.0
        )

        rms = (
            float(np.sqrt(np.mean(audio ** 2)))
            if number_of_samples > 0 else 0.0
        )

        clipped_samples = int(
            np.sum(np.abs(audio) >= CLIPPING_THRESHOLD)
        )

        # ChatGPT helped simplify the recording quality checks.
        if sample_rate != EXPECTED_SAMPLE_RATE:
            status_list.append("WRONG_SAMPLE_RATE")
            sample_rate_problem_count += 1

        if abs(duration - EXPECTED_DURATION) > DURATION_TOLERANCE:
            status_list.append("WRONG_DURATION")
            duration_problem_count += 1

        if rms < QUIET_RMS_THRESHOLD:
            status_list.append("TOO_QUIET")
            quiet_count += 1

        if clipped_samples >= MIN_CLIPPED_SAMPLES:
            status_list.append("POSSIBLE_CLIPPING")
            clipping_count += 1

        final_status = (
            ";".join(status_list) if status_list else "OK"
        )

        if status_list:
            flagged_files.append((
                wav_file, final_status,
                peak, rms, clipped_samples
            ))

        report_rows.append({
            "class": class_name,
            "file": wav_file.name,
            "sample_rate": sample_rate,
            "samples": number_of_samples,
            "duration_seconds": duration,
            "peak_amplitude": peak,
            "rms": rms,
            "clipped_samples": clipped_samples,
            "status": final_status
        })

    except Exception as error:
        read_error_count += 1
        final_status = "READ_ERROR: " + str(error)

        flagged_files.append((
            wav_file, final_status, "", "", ""
        ))

        report_rows.append({
            "class": class_name,
            "file": wav_file.name,
            "sample_rate": "",
            "samples": "",
            "duration_seconds": "",
            "peak_amplitude": "",
            "rms": "",
            "clipped_samples": "",
            "status": final_status
        })


# Save the original nine-column CSV report.
fieldnames = [
    "class", "file", "sample_rate", "samples",
    "duration_seconds", "peak_amplitude",
    "rms", "clipped_samples", "status"
]

with open(
    report_file, "w", newline="", encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report_rows)


# ChatGPT helped simplify the flagged recording summary.
print("\n==============================")
print("Flagged recordings")
print("==============================")

if not flagged_files:
    print("No suspicious recordings found.")
else:
    for wav_file, status, peak, rms, clipped in flagged_files:
        print(f"\n{wav_file.parent.name}/{wav_file.name}")
        print("Status:", status)

        if peak != "":
            print("Peak:", round(peak, 6))
            print("RMS:", round(rms, 6))
            print("Clipped samples:", clipped)


# Print the complete quality summary.
print("\n==============================")
print("Audio Quality Summary")
print("==============================")

print("Total WAV files:", len(wav_files))
print("Number of classes:", len(class_folders))
print("Classes with wrong file count:", len(class_count_problems))
print("Read errors:", read_error_count)
print("Wrong sample rate:", sample_rate_problem_count)
print("Wrong duration:", duration_problem_count)
print("Too quiet:", quiet_count)
print("Possible clipping:", clipping_count)
print("Total flagged files:", len(flagged_files))


# Show class-count problems.
if class_count_problems:
    print("\nClass count problems")

    for class_name, count in class_count_problems:
        print(f"{class_name}: {count} files")


print("\n==============================")
print("CSV report saved to:")
print(report_file)
print("==============================")
