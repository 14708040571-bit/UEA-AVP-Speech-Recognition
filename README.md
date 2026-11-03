# UEA Speech Recognition Project

## Overview

This project implements a speaker-dependent speech recognition system designed to classify 20 spoken names using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN).

The project was developed for the Audio-Visual Processing coursework at the University of East Anglia (UEA).

The system records and processes speech, extracts acoustic features, trains a CNN classifier, and predicts spoken names using either WAV files or live microphone input.

## Methods

- **Programming language:** Python 3.11
- **Development environment:** Spyder
- **Audio format:** 16 kHz, 2-second recordings
- **Recognition task:** 20 spoken-name classes
- **Feature extraction:** 13 MFCC coefficients
- **Classification model:** 2D Convolutional Neural Network (CNN)

### Dataset

The development and independent test datasets were collected separately.

- Development recordings: 400 (20 names × 20 recordings)
- Training subset for model selection: 320 recordings
- Validation subset for model selection: 80 recordings
- Independent test recordings: 100 (20 names × 5 recordings)

After model selection, the final CNN was trained on all 400 development recordings for 16 epochs.

### MFCC Feature Extraction

The MFCC extraction process consists of:

1. Loading WAV audio recordings.
2. Dividing audio into 25 ms frames with a 10 ms frame step.
3. Applying a Hamming window.
4. Computing the Fast Fourier Transform (FFT) and power spectrum.
5. Applying 26 triangular Mel filterbanks.
6. Computing logarithmic Mel filter energies.
7. Applying the Discrete Cosine Transform (DCT).
8. Retaining 13 MFCC coefficients.

The resulting MFCC feature representation has a shape of `(199, 13)` for each recording.

### CNN Model

The selected CNN configuration includes:

- Two convolutional layers with 16 and 32 filters
- Max-pooling layers
- A fully connected layer with 64 neurons
- Dropout rate: 0.3
- Optimiser: Adam
- Learning rate: 0.001
- Batch size: 16
- Final training epochs: 16

Additional experiments were conducted to investigate the effects of batch size, dropout and learning rate.

## Results

### Feature Comparison

Three acoustic feature representations were compared.

| Feature Type | Validation Accuracy |
|---|---:|
| Linear | 62.50% |
| Mel | 61.25% |
| MFCC | **81.25%** |

MFCC achieved the highest selected-model validation accuracy among the three feature representations.

### Independent Test Evaluation

| Evaluation | Result |
|---|---:|
| Selected-model validation accuracy | 81.25% |
| Independent test accuracy | 33.00% |
| Correct independent test predictions | 33 / 100 |
| Incorrect independent test predictions | 67 / 100 |

The selected-model validation accuracy was 81.25%, while the independent test accuracy was 33.00%, a difference of 48.25 percentage points.

This suggests that the final CNN had limited generalisation to separately collected recordings. Possible contributing factors include the relatively small development dataset, differences in recording conditions, and variations in pronunciation and speech timing.

Further investigation would be needed to determine the main cause of this performance gap. Validation and independent test results are reported separately because validation performance does not necessarily represent performance on unseen recordings.

## Project Files

All 14 Python scripts are located in the `src/` directory.

The repository contains **14 Python scripts**. Six scripts form the main implementation and evaluation workflow, while eight supporting scripts document data collection, dataset preparation and additional experiments.

### Core Scripts (6)

| File | Purpose |
|---|---|
| `feature_extraction.py` | Extracts MFCC features from audio |
| `feature_comparison.py` | Compares Linear, Mel and MFCC features |
| `train_model.py` | Trains and validates CNN models |
| `train_final_model.py` | Trains and saves the final CNN model |
| `evaluate_test.py` | Evaluates the final model on independent test recordings |
| `live_demo.py` | Performs recognition using microphone input or WAV files |

These six scripts represent the main feature extraction, model development, evaluation and demonstration stages.

### Supporting Scripts and Experiments (8)

| File | Purpose |
|---|---|
| `record_dataset.py` | Collects 400 development recordings |
| `record_test_dataset.py` | Collects 100 independent test recordings |
| `check_audio_quality.py` | Checks recorded audio quality |
| `prepare_dataset.py` | Prepares MFCC features for development data |
| `prepare_test_dataset.py` | Prepares MFCC features for independent testing |
| `experiment_batch_08.py` | Investigates batch size 8 |
| `experiment_dropout_05.py` | Investigates dropout rate 0.5 |
| `experiment_lr_0005.py` | Investigates learning rate 0.0005 |

These scripts are included to document the wider project workflow and experimental process.

### Additional Files

- `requirements.txt` — Python package dependencies
- `models/` — Trained CNN model and normalisation statistics
- `results/` — Evaluation results and project outputs
- `development_evidence/` — Screenshots documenting local development
- `presentation/` — Project presentation slides
- `.gitignore` — Excludes local recordings, datasets and unnecessary generated files

## Running the Project

The project was developed using Python 3.11 and Spyder.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Model and Data Files

The trained CNN model and normalisation statistics are included in the repository's `models/` directory.

The `live_demo.py` script first attempts to load these files from the repository. If they are unavailable, it attempts to load them from the original local project directory.

The script displays the model file path when the model is successfully loaded.

The saved-model demonstration can run without the original training dataset. However, reproducing the training experiments and independent test evaluation requires the original recordings and processed datasets.

These datasets are stored locally and are not included in this repository.

Some data preparation and training scripts still use the original local project path:

`~/Desktop/Assessment001`

These paths may need to be adjusted when running the project on another computer.

### Live Demonstration

From the repository directory, run:

```bash
python src/live_demo.py
```

The demonstration supports:

- Live microphone recording
- Prediction from WAV files
- Displaying prediction scores
- Displaying the top three predictions

Input WAV files must contain 2 seconds of audio sampled at 16 kHz.

The demonstration uses the saved CNN model and normalisation statistics without retraining the model.

When using microphone mode, the user can select an available audio input device. The device number depends on the computer and its connected audio devices.

The script records 2 seconds of speech after a short countdown and displays the predicted spoken name and the three highest-scoring predictions.

Prediction scores indicate the model's relative confidence and do not guarantee that the prediction is correct.

## Limitations and Future Improvements

### Current Limitations

- A relatively small development dataset
- Fixed 2-second input duration
- Sensitivity to pronunciation and recording conditions
- Lower independent test accuracy than validation accuracy
- Local file-path and audio-device dependencies in parts of the project

### Proposed Future Improvements

- Collecting more diverse speech recordings
- Applying audio data augmentation
- Investigating voice activity detection (VAD)
- Improving model generalisation
- Making file paths and microphone settings more portable

These are proposed future improvements and have not been implemented in the current project.

## Development and Version Control

The project was initially developed and tested locally using Spyder.

The GitHub repository was subsequently created to organise the existing project files and maintain a record of further changes.

The initial repository version contained the six main Python scripts. Additional scripts from the local project were later added to document data collection, dataset preparation and experimental work.

Later commits also included code review, documentation updates, formatting changes and fixes to improve usability, including improvements to model-path handling and microphone selection.

The original recordings and processed datasets remain stored locally rather than being uploaded to GitHub.

Development screenshots are available in [development_evidence](development_evidence/).

Presentation slides are available in [presentation](presentation/).

## AI Use Declaration

I used ChatGPT (OpenAI) to support my learning and development during this coursework. It assisted with understanding technical concepts such as MFCC and CNN, as well as Python coding, debugging, code review, documentation and presentation preparation.

I reviewed and tested the suggested changes, conducted the experiments, and take responsibility for the final implementation and results. I am able to explain and demonstrate my project independently.
---

**University of East Anglia | Audio-Visual Processing**
