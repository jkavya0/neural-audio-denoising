
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = PROJECT_ROOT / "data/dataset/parquet/data"
OUTPUT_DIR = PROJECT_ROOT / "data/dataset"

SPLITS = {
    "train": sorted(PARQUET_DIR.glob("train-*.parquet")),
    "test": sorted(PARQUET_DIR.glob("test-*.parquet")),
}


def extract_split(split_name, parquet_files):
    clean_dir = OUTPUT_DIR / split_name / "clean"
    noisy_dir = OUTPUT_DIR / split_name / "noisy"

    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0

    for parquet_file in parquet_files:
        print(f"Processing: {parquet_file.name}")

        parquet_table = pq.read_table(parquet_file)

        for record in parquet_table.to_pylist():
            recording_id = record["id"]

            clean_audio = sf.read(
                io.BytesIO(record["clean"]["bytes"]),
                dtype="float32"
            )[0]

            noisy_audio = sf.read(
                io.BytesIO(record["noisy"]["bytes"]),
                dtype="float32"
            )[0]

            if clean_audio.shape != noisy_audio.shape:
                raise ValueError(
                    f"Length mismatch for {recording_id}"
                )

            filename = f"{recording_id}.wav"

            sf.write(
                clean_dir / filename,
                clean_audio,
                16000,
                subtype="FLOAT"
            )

            sf.write(
                noisy_dir / filename,
                noisy_audio,
                16000,
                subtype="FLOAT"
            )

            extracted += 1

        print(f"Records extracted so far: {extracted}")

    print(f"\n{split_name.upper()} extraction complete: {extracted} pairs")


def main():
    for split_name, parquet_files in SPLITS.items():
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for {split_name}"
            )

        extract_split(split_name, parquet_files)


if __name__ == "__main__":
    main()
