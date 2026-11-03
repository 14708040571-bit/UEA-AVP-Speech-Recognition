# UEA AVP Speech Recognition

## Project Overview

This project develops a speaker-dependent speech recognition system to recognise 20 spoken names using MFCC features and a Convolutional Neural Network (CNN).

**Pipeline:** Audio → MFCC → CNN → Prediction

## Dataset

- 20 spoken names, recorded by one speaker
- 400 development recordings (320 training, 80 validation)
- 100 independent test recordings
- Audio: 2 seconds, 16 kHz

## Feature Extraction

MFCC features were implemented in Python using:

25 ms frames → Hamming window → 512-point FFT → Power Spectrum → 26 Mel filters → Log → DCT → 13 MFCC coefficients.

Frame step: 10 ms

**Feature output: 199 × 13**

## CNN Model

- Two Conv2D layers: 16 and 32 filters
- MaxPooling, Flatten and Dense (64 units)
- Dropout: 0.3
- Softmax: 20 classes
- Adam optimiser, learning rate 0.001
- Batch size: 16

## Results

| Feature | Validation Accuracy |
|---|---:|
| Linear | 62.50% |
| Mel | 61.25% |
| MFCC | 81.25% |

**Independent test accuracy: 33%.**

The results indicate limited generalisation to recordings from a separate session.

## Project Files

- `feature_extraction.py` — MFCC extraction
- `feature_comparison.py` — Feature comparison
- `train_model.py` — CNN training and validation
- `train_final_model.py` — Final model training
- `evaluate_test.py` — Independent testing
- `live_demo.py` — Speech recognition demo

## Future Improvements

Collect more recordings, improve recording consistency, and explore data augmentation and voice activity detection.

---

**University of East Anglia | Audio-Visual Processing Coursework**
