import pyarrow.parquet as pq

file_path = (
    "data/dataset/parquet/data/train-00000-of-00005.parquet"
)

table = pq.read_table(file_path)

print("Dataset schema:")
print(table.schema)

print("\nNumber of records:", table.num_rows)

# Inspect only the first record's metadata
record = table.slice(0, 1).to_pylist()[0]

print("\nRecording ID:", record["id"])

for column in ["clean", "noisy"]:
    audio = record[column]

    print(f"\n{column.upper()} AUDIO")
    print("Available fields:", audio.keys())
    print("Audio path:", audio.get("path"))
    print("Audio data type:", type(audio.get("bytes")))
    print("Audio byte size:", len(audio.get("bytes", b"")))