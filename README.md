
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
- MFCC: 81.25% validation accuracy
- Independent test accuracy: 33.00%

## Project Files
- `feature_extraction.py` — MFCC extraction
- `feature_comparison.py` — Feature comparison
- `train_model.py` — CNN training and validation
- `train_final_model.py` — Final CNN training
- `evaluate_test.py` — Independent testing
- `live_demo.py` — Microphone and WAV-file demonstration

## AI Use Declaration
Most of the project code was written by me. I used ChatGPT (OpenAI) to help understand technical concepts, assist with Python coding and debugging, and prepare project documentation and presentation materials.

For example, ChatGPT helped identify and remove a syntax error in an earlier version of `train_model.py`. I reviewed and tested the final code and results myself.

**University of East Anglia | Audio-Visual Processing**
