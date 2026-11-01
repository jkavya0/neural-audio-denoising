import soundfile as sf
import numpy as np


def load_audio(file_path, target_sample_rate=None):
    """
    Load a WAV audio file.

    Args:
        file_path: Path to the audio file.
        target_sample_rate: Optional target sampling rate.

    Returns:
        audio: Audio samples as a NumPy array.
        sample_rate: Sampling rate in Hz.
    """

    audio, sample_rate = sf.read(file_path)

    # Convert stereo audio to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Resample if required
    if target_sample_rate is not None:
        if sample_rate != target_sample_rate:
            from scipy.signal import resample_poly
            from math import gcd

            common = gcd(sample_rate, target_sample_rate)

            audio = resample_poly(
                audio,
                target_sample_rate // common,
                sample_rate // common
            )

            sample_rate = target_sample_rate

    return audio, sample_rate