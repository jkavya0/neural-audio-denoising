import numpy as np
import soundfile as sf
from pathlib import Path

sample_rate = 16000
duration = 3.0

t = np.arange(
    int(sample_rate * duration)
) / sample_rate

# Generate a combination of sine waves
audio = (
    0.5 * np.sin(2 * np.pi * 440 * t) +
    0.3 * np.sin(2 * np.pi * 880 * t)
)

output_dir = Path("data/clean")
output_dir.mkdir(parents=True, exist_ok=True)

sf.write(
    output_dir / "test_audio.wav",
    audio.astype(np.float32),
    sample_rate
)

print("Test audio generated successfully.")