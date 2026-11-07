
import numpy as np

from python.preprocessing import (
    load_audio,
    compute_stft,
    compute_istft,
    compute_magnitude,
    normalize_magnitude
)

audio, sample_rate = load_audio(
    "data/noisy/synthetic_speech_10dB.wav",
    target_sample_rate=16000
)

# Complex STFT
stft_matrix = compute_stft(audio)

# Magnitude spectrum
magnitude = compute_magnitude(stft_matrix)

# Temporary statistics for function validation only.
mean = float(np.mean(magnitude))
std = float(np.std(magnitude))

normalized = normalize_magnitude(magnitude, mean, std)

# Reconstruct using the original complex STFT.
reconstructed = compute_istft(stft_matrix)

min_length = min(len(audio), len(reconstructed))
mse = np.mean(
    (audio[:min_length] - reconstructed[:min_length]) ** 2
)

print("Sampling rate:", sample_rate)
print("Input samples:", len(audio))
print("STFT shape:", stft_matrix.shape)
print("Magnitude shape:", magnitude.shape)
print("Normalized shape:", normalized.shape)
print("Normalized mean:", np.mean(normalized))
print("Normalized standard deviation:", np.std(normalized))
print("Reconstruction MSE:", mse)

assert magnitude.shape == stft_matrix.shape
assert normalized.shape == magnitude.shape
assert np.isfinite(normalized).all()
assert np.allclose(np.mean(normalized), 0, atol=1e-5)
assert np.allclose(np.std(normalized), 1, atol=1e-5)
assert mse < 1e-8

print("All preprocessing tests passed.")
