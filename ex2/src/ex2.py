import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import add_watermark
import detect_watermark
from scipy.io import wavfile
from scipy.signal import stft


def plot_spectograms(audio_path):
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


def task1():
    add_watermark.add_watermark("audios/Task 1/task1.wav")
    og_arr, s_rate1 = librosa.load("audios/Task 1/task1.wav", sr=None)
    good_watermarked_arr, s_rate2 = librosa.load("audios/Task 1/good_task1result.wav", sr=None)
    bad_watermarked_arr, s_rate3 = librosa.load("audios/Task 1/bad_task1result.wav", sr=None)
    plot_spectograms("audios/Task 1/good_task1result.wav")
    plot_spectograms("audios/Task 1/bad_task1result.wav")


def task2():
    for i in range(9):
        detect_watermark.detect_peak_spacing(f"audios/Task 2/{i}_watermarked.wav")
        detect_watermark.display_spec_stft(f"audios/Task 2/{i}_watermarked.wav")


if __name__ == "__main__":
    task2()

    # Plot FFT magnitude for comparison
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.abs(original_fft), label='Original')
    # plt.plot(np.abs(watermarked_fft), label='Watermarked', alpha=0.75)
    # plt.legend()
    # plt.title("FFT Magnitude (Original vs Watermarked Audio)")
    # plt.xlabel("Frequency Bin")
    # plt.ylabel("Magnitude")
    # plt.grid(True)
    # plt.show()

