
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128

CLEAN_PATH = Path("data/clean/synthetic_speech.wav")
NOISY_DIR = Path("data/noisy")
OUTPUT_DIR = Path("results")

AUDIO_FILES = [
    ("Clean", CLEAN_PATH),
    ("20 dB", NOISY_DIR / "synthetic_speech_20dB.wav"),
    ("10 dB", NOISY_DIR / "synthetic_speech_10dB.wav"),
    ("5 dB", NOISY_DIR / "synthetic_speech_5dB.wav"),
    ("0 dB", NOISY_DIR / "synthetic_speech_0dB.wav"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = []

    for label, path in AUDIO_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        signal, sr = sf.read(path)

        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate in {path}")

        if signal.ndim > 1:
            raise ValueError(f"Expected mono audio: {path}")

        signals.append((label, signal.astype(np.float32)))

    reference = signals[0][1]

    for label, signal in signals[1:]:
        if len(signal) != len(reference):
            raise ValueError(f"Duration mismatch: {label}")

    # Common reference for comparable spectrogram levels.
    clean_stft = librosa.stft(
        reference,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )
    reference_magnitude = np.max(np.abs(clean_stft))

    max_amplitude = max(np.max(np.abs(signal)) for _, signal in signals)

    fig, axes = plt.subplots(
        len(signals), 2,
        figsize=(15, 16),
        constrained_layout=True
    )

    for row, (label, signal) in enumerate(signals):

        time = np.arange(len(signal)) / SAMPLE_RATE

        # Time-domain waveform.
        axes[row, 0].plot(time, signal, linewidth=0.5)
        axes[row, 0].set_title(f"{label} — Waveform")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].set_xlim(0, len(signal) / SAMPLE_RATE)
        axes[row, 0].set_ylim(-max_amplitude * 1.05, max_amplitude * 1.05)
        axes[row, 0].grid(True, alpha=0.3)

        # Frequency-domain spectrogram.
        stft = librosa.stft(
            signal,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window="hann"
        )

        magnitude = np.abs(stft)
        magnitude_db = 20 * np.log10(
            np.maximum(magnitude, 1e-10) / reference_magnitude
        )

        librosa.display.specshow(
            magnitude_db,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="linear",
            vmin=-80,
            vmax=0,
            ax=axes[row, 1]
        )

        axes[row, 1].set_title(f"{label} — Spectrogram")
        axes[row, 1].set_ylim(0, SAMPLE_RATE / 2)

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    fig.colorbar(
        axes[-1, 1].collections[0],
        ax=axes[:, 1].tolist(),
        label="Magnitude (dB relative to clean reference)"
    )

    fig.suptitle(
        "Clean and Noisy Speech Comparison",
        fontsize=16
    )

    output_path = OUTPUT_DIR / "clean_noisy_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Comparison saved: {output_path}")
    print(f"Number of recordings: {len(signals)}")
    print(f"Sampling rate: {SAMPLE_RATE} Hz")
    print(f"FFT size: {N_FFT}")
    print(f"Hop length: {HOP_LENGTH}")


if __name__ == "__main__":
    main()
