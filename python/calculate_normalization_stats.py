
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from python.preprocessing.stft import compute_stft, compute_magnitude


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_NOISY_DIR = PROJECT_ROOT / "data/dataset/train/noisy"
OUTPUT_FILE = PROJECT_ROOT / "python/models/normalization_stats.json"

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128


def main():
    audio_files = sorted(TRAIN_NOISY_DIR.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No training WAV files found in {TRAIN_NOISY_DIR}"
        )

    count = 0
    mean = 0.0
    m2 = 0.0

    for index, audio_file in enumerate(audio_files, start=1):
        audio, sample_rate = sf.read(audio_file, dtype="float32")

        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate: {audio_file}")

        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio: {audio_file}")

        if not np.all(np.isfinite(audio)):
            raise ValueError(f"Non-finite audio samples: {audio_file}")

        stft = compute_stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        magnitude = compute_magnitude(stft).astype(np.float64)

        batch_count = magnitude.size
        batch_mean = float(np.mean(magnitude))
        batch_m2 = float(np.sum((magnitude - batch_mean) ** 2))

        # Combine this recording's statistics with the running statistics.
        new_count = count + batch_count
        delta = batch_mean - mean

        m2 += batch_m2 + delta**2 * count * batch_count / new_count
        mean += delta * batch_count / new_count
        count = new_count

        if index % 1000 == 0 or index == len(audio_files):
            print(f"Processed {index}/{len(audio_files)} recordings")

    std = float(np.sqrt(m2 / count))

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid normalization statistics.")

    stats = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "mean": mean,
        "std": std,
        "number_of_recordings": len(audio_files),
        "number_of_values": count,
        "source_split": "train",
        "feature": "noisy_magnitude"
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=4)

    print("\nNormalization statistics calculated.")
    print(f"Mean: {mean:.8f}")
    print(f"Standard deviation: {std:.8f}")
    print(f"Total values: {count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
