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


def modulated_wave(t, A, f, mod_amp, mod_freq):
    """Model for the modulated sine wave"""
    return A * np.sin(2 * np.pi * f * t + mod_amp * np.cos(2 * np.pi * mod_freq * t))


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

    # Plot the original and reconstructed spectrograms
    plt.figure(figsize=(12, 8))

    # Original spectrogram
    plt.subplot(2, 1, 1)
    plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx)), shading='gouraud')
    plt.title('Original Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Magnitude (dB)')
    plt.ylim(0, 20e3)  # Focus on relevant frequency range

    # Reconstructed spectrogram
    plt.subplot(2, 1, 2)
    plt.pcolormesh(t_reconstructed, f_reconstructed, 20 * np.log10(np.abs(Zxx_reconstructed)), shading='gouraud')
    plt.title('Spectrogram of Extracted Wave (With Oscillations Preserved)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Magnitude (dB)')
    plt.ylim(0, 20e3)  # Same range for easy comparison

    plt.tight_layout()
    plt.show()

    # Fit the mathematical model to the reconstructed signal
    time = np.arange(len(x_reconstructed)) / sample_rate  # Time vector corresponding to the signal

    # Step 1: Extract carrier frequency
    carrier_frequency = 17.5e3 # Track the peak frequency across time

    # Step 2: Extract modulation frequency (using the oscillations in the carrier frequency)
    # Find frequency modulation peaks by checking periodic changes in the carrier frequency
    # We need to correctly estimate mod_freq, avoiding use of wrong time indices
    mod_freq = np.mean(np.diff(np.argmax(np.abs(Zxx_reconstructed), axis=0))) * sample_rate / len(t)

    # Step 3: Extract amplitude (since amplitude is not modulated, we just need the max amplitude of the wave)
    amplitude = np.max(np.abs(x_reconstructed))

    # print(f"Carrier Frequency: {carrier_frequency} Hz")
    # print(f"Modulation Frequency: {mod_freq} Hz")
    # print(f"Amplitude: {amplitude}")

    try:
        popt, _ = curve_fit(modulated_wave, time, x_reconstructed, p0=[1, carrier_frequency, 0, mod_freq])
    except Exception as e:
        print(f"Error fitting the curve: {e}")
        return

    # Extract fitted parameters
    A_fit, f_fit, mod_amp_fit, mod_freq_fit = popt

    # Print fitted parameters
    print(f"Fitted Parameters:\nAmplitude: {A_fit}, Carrier Frequency: {f_fit} Hz, "
          f"Modulation Amplitude: {mod_amp_fit}, Modulation Frequency: {mod_freq_fit} Hz")

    # Generate the fitted wave using the optimized parameters
    fitted_wave = modulated_wave(time, *popt)

    # # Plot the reconstructed wave and the fitted model
    # plt.figure(figsize=(12, 6))
    # plt.plot(time, x_reconstructed, label='Reconstructed Waveform')
    # plt.plot(time, fitted_wave, label='Fitted Waveform (Model)', linestyle='--')
    # plt.title('Reconstructed Waveform vs. Fitted Model')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()

    return fitted_wave, popt  # Return fitted wave and parameters


