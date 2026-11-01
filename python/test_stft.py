import numpy as np

from python.preprocessing.audio_io import load_audio
from python.preprocessing.stft import compute_stft, compute_istft


# Load test audio
audio, sample_rate = load_audio(
    "data/clean/test_audio.wav"
)

# Compute STFT
stft_matrix = compute_stft(audio)

# Reconstruct audio
reconstructed_audio = compute_istft(stft_matrix)

# Calculate reconstruction error
min_length = min(len(audio), len(reconstructed_audio))

error = audio[:min_length] - reconstructed_audio[:min_length]

mse = np.mean(error ** 2)

print("Sampling rate:", sample_rate)
print("Original samples:", len(audio))
print("STFT shape:", stft_matrix.shape)
print("Reconstructed samples:", len(reconstructed_audio))
print("Reconstruction MSE:", mse)