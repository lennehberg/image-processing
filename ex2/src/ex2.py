import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import add_watermark


def plot_spectograms(original_audio, watermarked_audio, sr):
    # Compute and display spectrograms
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    # original_audio = np.log(np.abs(original_audio) + 1)
    # watermarked_audio = np.log(np.abs(watermarked_audio) + 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max),
                             sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Original Audio Spectrogram")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(watermarked_audio)), ref=np.max),
                             sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Watermarked Audio Spectrogram")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    add_watermark.add_watermark("audios/Task 1/task1.wav")
    og_arr, s_rate1 = librosa.load("audios/Task 1/task1.wav", sr=None)
    watermarked_arr, s_rate2 = librosa.load("audios/Task 1/task1result.wav", sr=None)
    plot_spectograms(og_arr, watermarked_arr, s_rate1)

    # Compute FFT of original and watermarked audio
    original_fft = np.fft.fft(og_arr)
    watermarked_fft = np.fft.fft(watermarked_arr)

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

