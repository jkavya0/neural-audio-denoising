
import numpy as np
import librosa


def compute_stft(audio, n_fft=512, hop_length=128):
    """
    Compute the complex-valued STFT.

    Framing and Hann windowing are performed by librosa.
    """
    return librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann"
    )

    return stft_matrix


def compute_istft(stft_matrix, hop_length=128):
    """Reconstruct audio from its complex STFT."""
    return librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        window="hann"
    )


def compute_magnitude(stft_matrix):
    """Extract magnitude while retaining the original complex STFT."""
    return np.abs(stft_matrix).astype(np.float32)


def normalize_magnitude(magnitude, mean, std):
    """
    Standardize magnitude features using supplied training statistics.

    The same mean and standard deviation must be reused during inference.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive.")

    return ((magnitude - mean) / std).astype(np.float32)
