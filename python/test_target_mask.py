
import numpy as np

from python.preprocessing.audio_io import load_audio
from python.preprocessing.stft import compute_stft, compute_magnitude
from python.preprocessing.mask import compute_target_mask


def main():
    clean_path = "data/dataset/train/clean/p226_001.wav"
    noisy_path = "data/dataset/train/noisy/p226_001.wav"

    clean_audio, clean_sr = load_audio(clean_path, target_sample_rate=16000)
    noisy_audio, noisy_sr = load_audio(noisy_path, target_sample_rate=16000)

    assert clean_sr == noisy_sr == 16000
    assert clean_audio.shape == noisy_audio.shape

    clean_stft = compute_stft(clean_audio)
    noisy_stft = compute_stft(noisy_audio)

    clean_magnitude = compute_magnitude(clean_stft)
    noisy_magnitude = compute_magnitude(noisy_stft)

    target_mask = compute_target_mask(
        clean_magnitude,
        noisy_magnitude
    )

    print("Clean magnitude shape:", clean_magnitude.shape)
    print("Noisy magnitude shape:", noisy_magnitude.shape)
    print("Target mask shape:", target_mask.shape)
    print("Mask minimum:", target_mask.min())
    print("Mask maximum:", target_mask.max())
    print("Mask mean:", target_mask.mean())
    print("Mask data type:", target_mask.dtype)

    assert target_mask.shape == noisy_magnitude.shape
    assert target_mask.dtype == np.float32
    assert np.all(np.isfinite(target_mask))
    assert np.all(target_mask >= 0.0)
    assert np.all(target_mask <= 1.0)

    print("\nTarget mask tests passed.")


if __name__ == "__main__":
    main()
