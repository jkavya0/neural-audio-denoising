
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 16000
DURATION = 5.0


def generate_speech_like_signal():
    rng = np.random.default_rng(42)

    num_samples = int(SAMPLE_RATE * DURATION)
    t = np.arange(num_samples) / SAMPLE_RATE
    signal = np.zeros(num_samples, dtype=np.float64)

    # Approximate voiced and unvoiced speech segments.
    voiced_intervals = [
        (0.2, 0.8),
        (1.1, 1.7),
        (2.0, 2.6),
        (2.9, 3.5),
        (3.8, 4.5),
    ]

    # Create voiced segments using a harmonic source.
    f0 = 120 + 15 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    voiced = np.zeros_like(t)

    # Approximate vocal-tract resonances (formants).
    formants = [700, 1200, 2500]

    for harmonic in range(1, 35):
        frequency = harmonic * 120
        amplitude = sum(
            np.exp(-0.5 * ((frequency - formant) / 180) ** 2)
            for formant in formants
        )
        voiced += amplitude * np.sin(harmonic * phase) / harmonic

    # Normalize voiced component.
    voiced /= np.max(np.abs(voiced)) + 1e-8

    # Create filtered noise for unvoiced segments.
    noise = rng.normal(0, 1, num_samples)
    sos = butter(
        4,
        1500,
        btype="highpass",
        fs=SAMPLE_RATE,
        output="sos"
    )
    unvoiced = sosfilt(sos, noise)
    unvoiced /= np.max(np.abs(unvoiced)) + 1e-8

    # Assemble voiced and unvoiced regions.
    for start, end in voiced_intervals:
        mask = (t >= start) & (t < end)
        signal[mask] = voiced[mask]

    unvoiced_intervals = [
        (0.8, 1.1),
        (1.7, 2.0),
        (2.6, 2.9),
        (3.5, 3.8),
    ]

    for start, end in unvoiced_intervals:
        mask = (t >= start) & (t < end)
        signal[mask] = 0.35 * unvoiced[mask]

    # Apply a smooth amplitude envelope to avoid abrupt boundaries.
    fade_samples = int(0.02 * SAMPLE_RATE)

    for start, end in voiced_intervals + unvoiced_intervals:
        start_idx = int(start * SAMPLE_RATE)
        end_idx = int(end * SAMPLE_RATE)
        length = end_idx - start_idx
        fade = min(fade_samples, length // 2)

        if fade > 0:
            ramp = np.linspace(0, 1, fade)
            signal[start_idx:start_idx + fade] *= ramp
            signal[end_idx - fade:end_idx] *= ramp[::-1]

    # Peak normalization.
    signal = 0.8 * signal / (np.max(np.abs(signal)) + 1e-8)

    return signal.astype(np.float32)


if __name__ == "__main__":
    output_dir = Path("data/clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    speech = generate_speech_like_signal()

    output_path = output_dir / "synthetic_speech.wav"
    sf.write(output_path, speech, SAMPLE_RATE)

    print(f"Generated: {output_path}")
    print(f"Sampling rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {len(speech) / SAMPLE_RATE:.2f} seconds")
    print(f"Samples: {len(speech)}")
