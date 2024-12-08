import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, find_peaks


def display_spec_stft(audio_path):
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # Normalize the audio signal (if necessary)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    audio = audio / np.max(np.abs(audio))

    # Compute the STFT
    f, t, Zxx = stft(audio, sample_rate, nperseg=1024, noverlap=512)

    # Compute the magnitude of the STFT
    magnitude = np.abs(Zxx)

    # Normalize the magnitude to enhance brightness
    magnitude_normalized = magnitude / np.max(magnitude)

    # Apply log transformation (adjust scale as needed for brightness)
    magnitude_db = 20 * np.log10(magnitude_normalized + 1e-10) + 50  # Shift for better visibility

    # Plot the STFT (log magnitude)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, magnitude_db, shading='auto', cmap='magma')
    plt.title("Brightened Log-Transformed STFT Magnitude of Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()


def detect_peak_spacing(audio_path):
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # Normalize the audio signal (if necessary)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]

    # Compute the STFT
    f, t, Zxx = stft(audio, sample_rate, nperseg=1024, noverlap=512)
    print(np.abs(Zxx))
    # Compute the magnitude of the STFT
    magnitude = np.abs(Zxx)

    # Log transform to make features more prominent
    magnitude_db = 10 * np.log10(magnitude + 1e-10)  # Small value added to avoid log(0)

    # # Plot the STFT (magnitude)
    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(t, f, magnitude_db, shading='auto')
    # plt.title("Log-Transformed STFT Magnitude of Audio")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.colorbar(label="Magnitude (dB)")
    # plt.show()

    # Focus on the high-frequency range (18,000 Hz - 22,000 Hz)
    freq_range = (f >= 18000) & (f <= 22000)
    high_freq_magnitude = magnitude[freq_range, :]

    # Sum along time axis to get a "spectrum" of magnitudes for high-frequency range
    summed_magnitude = np.sum(high_freq_magnitude, axis=1)

    # Find peaks in the summed magnitude
    peaks, _ = find_peaks(summed_magnitude, distance=20)  # Distance controls peak separation

    # Extract the frequencies of the detected peaks
    peak_freqs = f[freq_range][peaks]

    # Calculate the spacings between consecutive peaks (in Hz)
    peak_spacings = np.diff(peak_freqs)

    # print("Detected peak frequencies:", peak_freqs)
    # print("Peak spacings (in Hz):", peak_spacings)

    return peak_freqs, peak_spacings


