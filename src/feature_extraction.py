
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.fft import dct


def extract_mfcc(audio_file, frame_length_ms=25,
                 frame_step_ms=10, nfft=512,
                 n_filters=26, n_mfcc=13):

    # Load WAV audio and use one channel.
    signal, fs = sf.read(audio_file)

    if signal.ndim > 1:
        signal = signal[:, 0]

    # Set the frame length and step.
    frame_length = int(fs * frame_length_ms / 1000)
    frame_step = int(fs * frame_step_ms / 1000)

    # Calculate the number of frames.
    n_frames = max(1, 1 + int(np.ceil(
        (len(signal) - frame_length) / frame_step
    )))

    # Pad the signal and divide it into frames.
    padded_length = (n_frames - 1) * frame_step + frame_length
    signal = np.pad(signal, (0, padded_length - len(signal)))

    indices = (
        np.arange(n_frames)[:, None] * frame_step
        + np.arange(frame_length)
    )

    frames = signal[indices]

    # Apply a Hamming window.
    frames = frames * np.hamming(frame_length)

    # ChatGPT assisted with understanding FFT and power spectra.
    spectrum = np.fft.rfft(frames, n=nfft, axis=1)
    power = np.abs(spectrum) ** 2 / nfft

    # ChatGPT helped explain Mel-frequency filterbank construction.
    high_mel = 2595 * np.log10(1 + (fs / 2) / 700)

    mel_points = np.linspace(0, high_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((nfft + 1) * hz_points / fs).astype(int)

    filterbank = np.zeros((n_filters, nfft // 2 + 1))

    # Construct triangular Mel filters.
    for m in range(1, n_filters + 1):
        left, centre, right = bins[m - 1:m + 2]

        if centre > left:
            filterbank[m - 1, left:centre] = (
                np.arange(left, centre) - left
            ) / (centre - left)

        if right > centre:
            filterbank[m - 1, centre:right] = (
                right - np.arange(centre, right)
            ) / (right - centre)

    # Compute Mel filter energies.
    energy = power @ filterbank.T

    # ChatGPT assisted with understanding log-energy processing.
    log_energy = np.log(
        np.maximum(energy, np.finfo(float).eps)
    )

    # ChatGPT explained DCT and the selection of 13 MFCCs.
    mfcc = dct(
        log_energy,
        type=2,
        axis=1,
        norm="ortho"
    )

    return mfcc[:, :n_mfcc]


# Test MFCC extraction using one training WAV file.
if __name__ == "__main__":

    project = Path.home() / "Desktop" / "Assessment001"
    wav_files = sorted((project / "data/train").rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError("No WAV files found.")

    mfcc = extract_mfcc(wav_files[0])

    print("Test file:", wav_files[0].name)
    print("MFCC shape:", mfcc.shape)
