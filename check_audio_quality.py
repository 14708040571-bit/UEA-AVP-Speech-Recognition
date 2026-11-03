from pathlib import Path
import csv

import numpy as np
import soundfile as sf


# ==========================================
# 1. Project paths
# ==========================================

project_folder = (
    Path.home()
    / "Desktop"
    / "Assessment001"
)

train_folder = (
    project_folder
    / "data"
    / "train"
)

results_folder = (
    project_folder
    / "results"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)

report_file = (
    results_folder
    / "audio_quality_report.csv"
)


# ==========================================
# 2. Expected recording settings
# ==========================================

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_DURATION = 2.0

# 允许时长有一点点误差
DURATION_TOLERANCE = 0.05

# 如果 RMS 太低，说明录音可能过静
QUIET_RMS_THRESHOLD = 0.005

# 非常接近数字音频最大值，可能发生 clipping
CLIPPING_THRESHOLD = 0.999

# 至少有这么多个样本接近满幅才标记 clipping
MIN_CLIPPED_SAMPLES = 5

EXPECTED_FILES_PER_CLASS = 20


# ==========================================
# 3. Find all WAV files
# ==========================================

wav_files = sorted(
    train_folder.glob("*/*.wav")
)


print()
print("==============================")
print("Audio Quality Check")
print("==============================")

print("Training folder:")
print(train_folder)

print()
print(
    "Total WAV files found:",
    len(wav_files)
)


# ==========================================
# 4. Count files in each class
# ==========================================

class_folders = sorted(
    [
        folder
        for folder in train_folder.iterdir()
        if folder.is_dir()
    ]
)


print()
print("==============================")
print("Files per class")
print("==============================")


class_count_problems = []


for folder in class_folders:

    number_of_files = len(
        list(
            folder.glob("*.wav")
        )
    )

    print(
        folder.name,
        ":",
        number_of_files
    )

    if (
        number_of_files
        != EXPECTED_FILES_PER_CLASS
    ):
        class_count_problems.append(
            (
                folder.name,
                number_of_files
            )
        )


# ==========================================
# 5. Check every WAV file
# ==========================================

report_rows = []

read_error_count = 0
sample_rate_problem_count = 0
duration_problem_count = 0
quiet_count = 0
clipping_count = 0

flagged_files = []


for wav_file in wav_files:

    class_name = wav_file.parent.name

    status_list = []

    try:

        audio, sample_rate = sf.read(
            wav_file,
            dtype="float32"
        )


        # ----------------------------------
        # Stereo -> mono
        # ----------------------------------

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1
            )


        number_of_samples = len(audio)


        # ----------------------------------
        # Duration
        # ----------------------------------

        duration = (
            number_of_samples
            / sample_rate
        )


        # ----------------------------------
        # Peak amplitude
        # ----------------------------------

        if number_of_samples > 0:

            peak = float(
                np.max(
                    np.abs(audio)
                )
            )

        else:

            peak = 0.0


        # ----------------------------------
        # RMS
        # ----------------------------------

        if number_of_samples > 0:

            rms = float(
                np.sqrt(
                    np.mean(
                        audio ** 2
                    )
                )
            )

        else:

            rms = 0.0


        # ----------------------------------
        # Count clipped samples
        # ----------------------------------

        clipped_samples = int(
            np.sum(
                np.abs(audio)
                >= CLIPPING_THRESHOLD
            )
        )


        # ----------------------------------
        # Check sample rate
        # ----------------------------------

        if (
            sample_rate
            != EXPECTED_SAMPLE_RATE
        ):

            status_list.append(
                "WRONG_SAMPLE_RATE"
            )

            sample_rate_problem_count += 1


        # ----------------------------------
        # Check duration
        # ----------------------------------

        if (
            abs(
                duration
                - EXPECTED_DURATION
            )
            > DURATION_TOLERANCE
        ):

            status_list.append(
                "WRONG_DURATION"
            )

            duration_problem_count += 1


        # ----------------------------------
        # Check quiet recordings
        # ----------------------------------

        if (
            rms
            < QUIET_RMS_THRESHOLD
        ):

            status_list.append(
                "TOO_QUIET"
            )

            quiet_count += 1


        # ----------------------------------
        # Check clipping
        # ----------------------------------

        if (
            clipped_samples
            >= MIN_CLIPPED_SAMPLES
        ):

            status_list.append(
                "POSSIBLE_CLIPPING"
            )

            clipping_count += 1


        # ----------------------------------
        # Final status
        # ----------------------------------

        if len(status_list) == 0:

            final_status = "OK"

        else:

            final_status = ";".join(
                status_list
            )

            flagged_files.append(
                (
                    wav_file,
                    final_status,
                    peak,
                    rms,
                    clipped_samples
                )
            )


        report_rows.append(
            {
                "class": class_name,
                "file": wav_file.name,
                "sample_rate": sample_rate,
                "samples": number_of_samples,
                "duration_seconds": duration,
                "peak_amplitude": peak,
                "rms": rms,
                "clipped_samples": clipped_samples,
                "status": final_status
            }
        )


    except Exception as error:

        read_error_count += 1

        final_status = (
            "READ_ERROR: "
            + str(error)
        )

        flagged_files.append(
            (
                wav_file,
                final_status,
                "",
                "",
                ""
            )
        )

        report_rows.append(
            {
                "class": class_name,
                "file": wav_file.name,
                "sample_rate": "",
                "samples": "",
                "duration_seconds": "",
                "peak_amplitude": "",
                "rms": "",
                "clipped_samples": "",
                "status": final_status
            }
        )


# ==========================================
# 6. Save CSV report
# ==========================================

fieldnames = [
    "class",
    "file",
    "sample_rate",
    "samples",
    "duration_seconds",
    "peak_amplitude",
    "rms",
    "clipped_samples",
    "status"
]


with open(
    report_file,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        report_rows
    )


# ==========================================
# 7. Print flagged files
# ==========================================

print()
print("==============================")
print("Flagged recordings")
print("==============================")


if len(flagged_files) == 0:

    print(
        "No suspicious recordings found."
    )

else:

    for (
        wav_file,
        status,
        peak,
        rms,
        clipped_samples
    ) in flagged_files:

        print()

        print(
            wav_file.parent.name,
            "/",
            wav_file.name
        )

        print(
            "Status:",
            status
        )

        if peak != "":

            print(
                "Peak:",
                round(
                    peak,
                    6
                )
            )

            print(
                "RMS:",
                round(
                    rms,
                    6
                )
            )

            print(
                "Clipped samples:",
                clipped_samples
            )


# ==========================================
# 8. Final summary
# ==========================================

print()
print("==============================")
print("Audio Quality Summary")
print("==============================")

print(
    "Total WAV files:",
    len(wav_files)
)

print(
    "Number of classes:",
    len(class_folders)
)

print(
    "Classes with wrong file count:",
    len(class_count_problems)
)

print(
    "Read errors:",
    read_error_count
)

print(
    "Wrong sample rate:",
    sample_rate_problem_count
)

print(
    "Wrong duration:",
    duration_problem_count
)

print(
    "Too quiet:",
    quiet_count
)

print(
    "Possible clipping:",
    clipping_count
)

print(
    "Total flagged files:",
    len(flagged_files)
)


# ==========================================
# 9. Print class-count problems
# ==========================================

if len(class_count_problems) > 0:

    print()
    print("==============================")
    print("Class count problems")
    print("==============================")

    for (
        class_name,
        count
    ) in class_count_problems:

        print(
            class_name,
            ":",
            count,
            "files"
        )


# ==========================================
# 10. Report location
# ==========================================

print()
print("==============================")
print("CSV report saved to:")
print(report_file)
print("==============================")