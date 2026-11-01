import numpy as np
import librosa


def compute_stft(audio, n_fft=512, hop_length=128):
    """
    Compute the Short-Time Fourier Transform.

    Args:
        audio: Input audio signal.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.

    Returns:
        Complex-valued STFT matrix.
    """

    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann"
    )

    return stft_matrix


def compute_istft(stft_matrix, hop_length=128):
    """
    Reconstruct audio from its STFT representation.
    """

    audio = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        window="hann"
    )

    return audio