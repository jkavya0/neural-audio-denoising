
import numpy as np
import soundfile as sf
from pathlib import Path

SNR_LEVELS = [20, 10, 5, 0]


def add_awgn(clean_signal, snr_db, rng):
    """Add white Gaussian noise at a specified SNR."""

    signal_power = np.mean(clean_signal ** 2)

    if signal_power <= 0:
        raise ValueError("Clean signal must have non-zero power.")

    noise = rng.normal(0.0, 1.0, len(clean_signal))
    noise_power = np.mean(noise ** 2)

    snr_linear = 10 ** (snr_db / 10)
    required_noise_power = signal_power / snr_linear

    noise *= np.sqrt(required_noise_power / noise_power)

    noisy_signal = clean_signal + noise

    return noisy_signal.astype(np.float32)


def calculate_snr(clean_signal, noisy_signal):
    """Calculate measured SNR from clean and noisy signals."""

    noise = noisy_signal - clean_signal

    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


if __name__ == "__main__":
    clean_path = Path("data/clean/synthetic_speech.wav")
    output_dir = Path("data/noisy")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_signal, sample_rate = sf.read(clean_path)
    clean_signal = clean_signal.astype(np.float32)

    rng = np.random.default_rng(42)

    for snr_db in SNR_LEVELS:
        noisy_signal = add_awgn(clean_signal, snr_db, rng)

        measured_snr = calculate_snr(clean_signal, noisy_signal)

        output_path = output_dir / f"synthetic_speech_{snr_db}dB.wav"

        sf.write(
            output_path,
            noisy_signal,
            sample_rate,
            subtype="FLOAT"
        )

        print(
            f"Target SNR: {snr_db:>2} dB | "
            f"Measured SNR: {measured_snr:6.2f} dB | "
            f"Saved: {output_path}"
        )
