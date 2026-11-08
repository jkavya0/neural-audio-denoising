
import io
import soundfile as sf
import pyarrow.parquet as pq

file_path = (
    "data/dataset/parquet/data/train-00000-of-00005.parquet"
)

table = pq.read_table(file_path)
record = table.slice(0, 1).to_pylist()[0]

for column in ["clean", "noisy"]:
    audio_bytes = record[column]["bytes"]

    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    print(f"\n{column.upper()} AUDIO")
    print("Sample rate:", sample_rate)
    print("Shape:", audio.shape)
    print("Duration:", len(audio) / sample_rate, "seconds")
    print("Data type:", audio.dtype)
    print("Peak amplitude:", abs(audio).max())

    assert sample_rate == 16000
    assert audio.ndim == 1
    assert len(audio) > 0

print("\nAudio decoding tests passed.")
