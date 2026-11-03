# UEA Speech Recognition Project

## Overview

A speaker-dependent speech recognition system for recognising 20 spoken names using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN).

This project was developed for the Audio-Visual Processing coursework at the University of East Anglia (UEA).

The system processes short speech recordings, extracts acoustic features, trains a CNN classifier, and predicts spoken names using either recorded WAV files or live microphone input.

## Methods

- **Programming language:** Python 3.11
- **Development environment:** Spyder
- **Audio format:** 2-second recordings at 16 kHz
- **Recognition task:** Classification of 20 spoken names
- **Feature extraction:** 13 MFCC coefficients
- **Classification model:** 2D Convolutional Neural Network (CNN)

### Dataset

- 400 development recordings, with 20 recordings per name.
- The development dataset was split into 320 training and 80 validation recordings for model selection.
- The final CNN was retrained on all 400 development recordings for 16 epochs.
- 100 independent test recordings, with 5 recordings per name.

### Feature Extraction

The MFCC extraction process includes:

1. Loading WAV audio recordings.
2. Dividing the audio into 25 ms frames with a 10 ms frame step.
3. Applying a Hamming window.
4. Computing the Fast Fourier Transform (FFT) and power spectrum.
5. Applying 26 triangular Mel filterbanks.
6. Computing logarithmic Mel filter energies.
7. Applying the Discrete Cosine Transform (DCT).
8. Retaining the first 13 MFCC coefficients.

The resulting MFCC features are used as input to the CNN classifier.

### Model Training

A 2D CNN was used to learn local patterns from the extracted speech features.

The final model configuration includes:

- Two convolutional layers with 16 and 32 filters.
- Max-pooling layers.
- A fully connected layer with 64 neurons.
- Dropout rate of 0.3.
- Adam optimiser with a learning rate of 0.001.
- Batch size of 16.
- Final training for 16 epochs.

Additional experiments investigated different training hyperparameters, including batch size, dropout rate and learning rate.

## Results

### Feature Comparison

| Feature Type | Validation Accuracy |
|---|---|
| Linear | 62.50% |
| Mel | 61.25% |
| MFCC | 81.25% |

MFCC features achieved the highest selected-model validation accuracy among the three feature representations.

### Independent Testing

- **Selected-model MFCC validation accuracy:** 81.25%
- **Independent test accuracy:** 33.00%

The independent test accuracy was considerably lower than the validation accuracy, indicating limitations in generalisation to recordings collected separately from the development dataset.

Possible contributing factors include the limited amount of training data and differences in pronunciation, recording conditions and speech timing.

## Project Files

### Data Collection and Preparation

- `record_dataset.py` — Records 400 training audio samples across 20 spoken-name classes.
- `record_test_dataset.py` — Records 100 independent test audio samples.
- `check_audio_quality.py` — Checks the quality of recorded audio.
- `prepare_dataset.py` — Extracts MFCC features and prepares the training dataset.
- `prepare_test_dataset.py` — Extracts MFCC features and prepares the independent test dataset.

### Feature Extraction and Comparison

- `feature_extraction.py` — Extracts MFCC features from audio recordings.
- `feature_comparison.py` — Compares Linear, Mel and MFCC features.

### Model Training and Experiments

- `train_model.py` — Trains and validates the CNN model.
- `experiment_batch_08.py` — Experiments with a batch size of 8.
- `experiment_dropout_05.py` — Experiments with a dropout rate of 0.5.
- `experiment_lr_0005.py` — Experiments with a learning rate of 0.0005.
- `train_final_model.py` — Trains the final CNN using all 400 development recordings.

### Evaluation and Demonstration

- `evaluate_test.py` — Evaluates the final model using independent test recordings.
- `live_demo.py` — Demonstrates speech recognition using microphone input or WAV files.

### Additional Files and Folders

- `requirements.txt` — Lists the Python packages required for the project.
- `models/` — Contains the trained CNN model and normalisation statistics.
- `results/` — Contains project results and related outputs.
- `development_evidence/` — Contains screenshots documenting local development.
- `.gitignore` — Excludes personal recordings, local datasets and unnecessary generated files from version control.

## Running the Project

The project was developed using Python 3.11 and Spyder.

Required Python packages are listed in `requirements.txt`.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

The trained CNN model and normalisation statistics are available in the `models/` folder.

The original speech recordings and processed datasets are stored locally and are not included in this repository.

The current scripts use the local project path `~/Desktop/Assessment001`. This path must be adjusted when running the project on another computer.

### Live Demonstration

The `live_demo.py` script supports:

- Recording speech using a microphone.
- Predicting spoken names from WAV files.
- Displaying prediction scores.
- Showing the three highest-scoring predictions.

Input WAV files must be 2 seconds long with a sampling rate of 16 kHz.

The demonstration uses the saved CNN model and normalisation statistics without retraining the model.

## Limitations and Future Improvements

The current system has several limitations:

- The development dataset contains a relatively small number of recordings.
- The system uses a fixed 2-second input duration.
- Recognition performance may be affected by differences in pronunciation and recording conditions.
- Independent test accuracy is substantially lower than validation accuracy.
- Some scripts depend on local file paths and audio-device configurations.

Possible future improvements include:

- Collecting more diverse training recordings.
- Applying audio data augmentation.
- Investigating voice activity detection (VAD).
- Improving model generalisation.
- Making file paths and recording-device settings more portable.

These improvements are proposed future work and have not been implemented in the current system.

## Development and Version Control

I developed and tested my project locally using Spyder before creating this GitHub repository.

The private GitHub repository was created later to organise my existing code and track further changes.

The repository includes the main implementation, data preparation scripts, recording utilities and experimental scripts to document the project development process.

The original speech recordings and processed datasets are kept locally rather than uploaded to GitHub.

Local development evidence: [View screenshots](development_evidence/)

## AI Use Declaration

I used ChatGPT (OpenAI) to support my learning and assist with Python coding and debugging during this project.

Specifically, ChatGPT helped me understand MFCC feature extraction, Mel filterbanks, the Discrete Cosine Transform (DCT), feature normalisation, CNN training, validation and independent testing. It also helped identify and correct a syntax error in an earlier version of `train_model.py`.

I also used ChatGPT to assist with reviewing and editing parts of the repository documentation and preparing presentation materials. I have disclosed these uses for transparency.

I reviewed and tested the code and experimental results. I take responsibility for understanding and explaining the implementation, design decisions and evaluation results.

I will not use generative AI tools during the bench demo.

---

**University of East Anglia | Audio-Visual Processing**
