
import numpy as np


def compute_target_mask(
    clean_magnitude,
    noisy_magnitude,
    epsilon=1e-8
):
    """
    Calculate the ideal ratio spectral mask.

    Args:
        clean_magnitude: Magnitude spectrum of clean speech.
        noisy_magnitude: Magnitude spectrum of noisy speech.
        epsilon: Small value to prevent division by zero.

    Returns:
        Target spectral mask with values in [0, 1].
    """

    if clean_magnitude.shape != noisy_magnitude.shape:
        raise ValueError("Clean and noisy spectra must have matching shapes.")

    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")

    mask = clean_magnitude / (noisy_magnitude + epsilon)

    mask = np.clip(mask, 0.0, 1.0)

    return mask.astype(np.float32)
