import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.optimize import curve_fit


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


def extract_watermark(audio_path):
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # # Normalize the audio signal (if necessary)
    # if audio.ndim > 1:
    #     audio = audio.mean(axis=1)  # Convert stereo to mono
    audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]

    # Compute the STFT
    f, t, Zxx = stft(audio, sample_rate, nperseg=1024)

    # Extract the frequency band around 17.5 kHz
    freq_band = (f > 16e3) & (f < 20e3)  # Slightly wider band to include oscillations
    Zxx_filtered = np.zeros_like(Zxx)  # Create a filtered STFT
    Zxx_filtered[freq_band, :] = Zxx[freq_band, :]  # Retain only the desired band

    # Reconstruct the signal from the filtered STFT
    _, x_reconstructed = istft(Zxx_filtered, sample_rate)

    # Compute the spectrogram of the reconstructed signal
    f_reconstructed, t_reconstructed, Zxx_reconstructed = stft(x_reconstructed, sample_rate, nperseg=1024)

    # # Plot the original and reconstructed spectrograms
    # plt.figure(figsize=(12, 8))
    #
    # # Original spectrogram
    # plt.subplot(2, 1, 1)
    # plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)), shading='gouraud')
    # plt.title(f'Original Spectrogram of {audio_path}')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.colorbar(label='Magnitude (dB)')
    # plt.ylim(0, 20e3)  # Focus on relevant frequency range
    #
    # # Reconstructed spectrogram
    # plt.subplot(2, 1, 2)
    # plt.pcolormesh(t_reconstructed, f_reconstructed, 20 * np.log10(np.abs(Zxx_reconstructed)), shading='gouraud')
    # plt.title('Spectrogram of Extracted Wave (With Oscillations Preserved)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.colorbar(label='Magnitude (dB)')
    # plt.ylim(0, 20e3)  # Same range for easy comparison
    #
    # plt.tight_layout()
    # plt.show()




