from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mediapy as media
import librosa
import soundfile as sf


def __get_high_freq_indices(high_freq_start, high_freq_end, frequencies):
    return np.where((frequencies >= high_freq_start) & (frequencies <= high_freq_end))


def __add_watermark_to_fft(aud_fft, frequencies, high_freq_indices):
    # create a sinusoidal signal with a small amplitude and add to fft
    watermark_magnitude = 1
    watermark_signal = watermark_magnitude * np.sin(2 * np.pi * frequencies[high_freq_indices])

    aud_fft[high_freq_indices] += watermark_signal
    return aud_fft


def __inverse_fft(aud_fft):
    # perform inverse fft to get the signal back in time domain
    watermarked_arr = np.fft.ifft(aud_fft)
    # remove negative values
    watermarked_arr = np.real(watermarked_arr)
    return watermarked_arr


def __add_frequencies(aud_arr, s_rate):
    high_freq_start = 20000
    high_freq_end = 22000
    # run the fast fourier transform on the audio file to get
    # sine-cosine representation of the function related to the
    # audio array
    aud_fft = np.fft.fft(aud_arr)
    # get the frequencies from the fft
    frequencies = np.fft.fftfreq(len(aud_arr), d=1/s_rate)
    # get the indices of the high frequencies
    high_freq_indices = __get_high_freq_indices(high_freq_start, high_freq_end, frequencies)
    aud_fft = __add_watermark_to_fft(aud_fft, frequencies, high_freq_indices)
    watermarked = __inverse_fft(aud_fft)
    return watermarked


def add_watermark(audio_file_path):
    """
    Add an unheard watermark to an audio file
    :param audio_file_path: path to audio file
    :return: write out audio file to /audios/results
    """
    # open audio file using librosa and get sampling rate
    aud_arr, s_rate = librosa.load(audio_file_path, sr=None)
    min_s_rate = 44000
    out_file_path = "audios/Task 1/task1result.wav"
    # check that sampling rate is high enough to add frequencies at the
    # ~20kHz mark
    if s_rate < min_s_rate:
        print("Error: Sampling rate too low!")
        return

    # add high range frequencies to audio array
    watermarked_aud = __add_frequencies(aud_arr, s_rate)

    # write to task1result.wav
    sf.write(out_file_path, watermarked_aud, s_rate)
