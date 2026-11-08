
from pathlib import Path
import soundfile as sf

ROOT = Path("data/dataset")

for split in ["train", "test"]:
    clean_dir = ROOT / split / "clean"
    noisy_dir = ROOT / split / "noisy"

    clean_files = sorted(clean_dir.glob("*.wav"))
    noisy_files = sorted(noisy_dir.glob("*.wav"))

    clean_names = {f.name for f in clean_files}
    noisy_names = {f.name for f in noisy_files}

    assert clean_names == noisy_names, (
        f"Pairing mismatch in {split}"
    )

    for clean_file in clean_files:
        noisy_file = noisy_dir / clean_file.name

        clean_audio, clean_sr = sf.read(clean_file)
        noisy_audio, noisy_sr = sf.read(noisy_file)

        assert clean_sr == noisy_sr == 16000
        assert clean_audio.shape == noisy_audio.shape

    print(f"{split.upper()} validation passed")
    print(f"Clean files: {len(clean_files)}")
    print(f"Noisy files: {len(noisy_files)}")

print("\nDataset validation completed.")
