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


def task1():
    add_watermark.add_watermark("audios/Task 1/task1.wav")
    og_arr, s_rate1 = librosa.load("audios/Task 1/task1.wav", sr=None)
    good_watermarked_arr, s_rate2 = librosa.load("audios/Task 1/good_task1result.wav", sr=None)
    bad_watermarked_arr, s_rate3 = librosa.load("audios/Task 1/bad_task1result.wav", sr=None)
    plot_spectograms(og_arr, good_watermarked_arr, s_rate1)
    plot_spectograms(og_arr, bad_watermarked_arr, s_rate1)


def task2():
    for i in range(9):
        add_watermark.add_watermark(f"audios/Task 2/{i}_watermarked.wav", index=i, task_num=2)
        aud1, s_rate1 = librosa.load(f"audios/Task 2/{i}_watermarked.wav", sr=40000)
        good_watermarked_arr, s_rate2 = librosa.load(f"audios/Task 2/good_task{i}result.wav", sr=None)
        plot_spectograms(aud1, good_watermarked_arr, 40000)


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

