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

The independent test accuracy was substantially lower than the validation accuracy, indicating limited generalisation to separately collected recordings.

Possible contributing factors include the small development dataset, differences in pronunciation and recording conditions, and variations in speech timing.

These results are reported separately because validation performance does not necessarily represent performance on unseen recordings.

## Project Files

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

These scripts represent the main feature extraction, model development, evaluation and demonstration stages.

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

These files are included to document the wider project workflow and experimental process.

### Additional Files

- `requirements.txt` — Python package dependencies
- `models/` — Trained CNN model and normalisation statistics
- `results/` — Evaluation results and project outputs
- `development_evidence/` — Screenshots of local development
- `.gitignore` — Excludes local recordings, datasets and unnecessary generated files

## Running the Project

The project was developed using Python 3.11 and Spyder.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The saved CNN model and normalisation statistics are available in the `models/` directory.

### Live Demonstration

From the repository directory, run:

```bash
python live_demo.py
```

The demonstration supports:

- Live microphone recording
- Prediction from WAV files
- Displaying prediction scores
- Displaying the top three predictions

The system uses 16 kHz, 2-second audio recordings.

The demonstration loads a saved CNN model and normalisation statistics. Retraining is not required for inference.

### Local File Paths and Data

The original WAV recordings and processed datasets are stored locally and are not included in this repository.

Some scripts use the local project directory:

`~/Desktop/Assessment001`

These paths may need to be adjusted when running the project on another computer.

The Live Demo also supports loading saved model files from the repository, although its existing local model-path priority should be checked when using different model versions.

## Limitations and Future Improvements

### Current Limitations

- A relatively small development dataset
- Fixed 2-second input duration
- Sensitivity to pronunciation and recording conditions
- Lower independent test accuracy than validation accuracy
- Local file-path and audio-device dependencies

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

Later commits also included code review, documentation updates, formatting changes and fixes to improve usability.

The original recordings and processed datasets remain stored locally rather than being uploaded to GitHub.

Development screenshots are available in [development_evidence](development_evidence/).

## AI Use Declaration

I used ChatGPT (OpenAI) as an AI-assisted learning and development tool during this coursework.

ChatGPT assisted me with understanding technical concepts, including MFCC feature extraction, Mel filterbanks, DCT, feature normalisation, CNN training, validation and independent testing.

I also used ChatGPT to help review, debug and revise parts of my Python scripts. This included identifying a syntax error in an earlier version of `train_model.py`, reviewing code structure, and assisting with file-path and microphone-input issues.

ChatGPT also assisted with documentation editing, GitHub organisation and the preparation of presentation materials.

I reviewed and tested the implementation and recorded the experimental results from my local development environment.

I take responsibility for understanding and explaining the implementation, the experimental decisions and the reported results.

During the assessed bench demonstration, I intend to explain and demonstrate the project independently, in accordance with the module's assessment requirements.

---

**University of East Anglia | Audio-Visual Processing**
