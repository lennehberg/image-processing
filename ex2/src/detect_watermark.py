import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft


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
