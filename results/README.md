
# UEA Speech Recognition Project

## Overview
A speaker-dependent speech recognition system for recognising 20 spoken names using MFCC features and a CNN.

## Methods
- Python, MFCC (13 coefficients), and a 2D CNN
- 400 development recordings (320 training, 80 validation)
- 100 independent test recordings
- 2-second audio recordings at 16 kHz

## Results
- Linear: 62.50% validation accuracy
- Mel: 61.25% validation accuracy
- MFCC: 81.25% selected-model validation accuracy
- Independent test accuracy: 33.00%

## Project Files
- `feature_extraction.py` — MFCC extraction
- `feature_comparison.py` — Feature comparison
- `train_model.py` — CNN training and validation
- `train_final_model.py` — Final CNN training
- `evaluate_test.py` — Independent testing
- `live_demo.py` — Microphone and WAV-file demonstration

## Running the Project
The project was developed using Python 3.11 and Spyder.

Required Python packages are listed in `requirements.txt`.

The trained CNN model and normalisation statistics are available in the `models/` folder.

The original speech recordings and processed datasets are stored locally and are not included in this repository.

The current scripts use the local project path `~/Desktop/Assessment001`. This path must be adjusted when running the project on another computer.

`live_demo.py` supports microphone recording and WAV-file prediction. Input WAV files must be 2 seconds long at a sampling rate of 16 kHz.

## Development and Version Control
I developed and tested my project locally using Spyder before creating this GitHub repository.

The private GitHub repository was created later to organise my existing code and track further changes.

Local development evidence: [View screenshots](development_evidence/)

## AI Use Declaration
Most of the project code was written by me. I used ChatGPT (OpenAI) to help understand technical concepts, assist with Python coding and debugging, and prepare project documentation and presentation materials.

For example, ChatGPT helped identify and remove a syntax error in an earlier version of `train_model.py`. I reviewed and tested the final code and results myself.

**University of East Anglia | Audio-Visual Processing**
