
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from python.datasets.speech_dataset import SpeechDenoisingDataset


ROOT = Path(__file__).resolve().parents[1]

dataset = SpeechDenoisingDataset(
    clean_dir=ROOT / "data/dataset/train/clean",
    noisy_dir=ROOT / "data/dataset/train/noisy",
    normalization_file=ROOT / "python/models/normalization_stats.json",
    segment_seconds=2.0,
    sample_rate=16000
)

print("Dataset size:", len(dataset))

# Test one sample.
input_tensor, target_tensor = dataset[0]

print("\nSingle sample:")
print("Input shape:", tuple(input_tensor.shape))
print("Target shape:", tuple(target_tensor.shape))
print("Input dtype:", input_tensor.dtype)
print("Target dtype:", target_tensor.dtype)

assert input_tensor.shape == (1, 257, 251)
assert target_tensor.shape == (1, 257, 251)
assert input_tensor.dtype == torch.float32
assert target_tensor.dtype == torch.float32
assert torch.isfinite(input_tensor).all()
assert torch.isfinite(target_tensor).all()
assert target_tensor.min() >= 0.0
assert target_tensor.max() <= 1.0

# Test batching.
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

batch_input, batch_target = next(iter(loader))

print("\nFirst batch:")
print("Input batch shape:", tuple(batch_input.shape))
print("Target batch shape:", tuple(batch_target.shape))

assert batch_input.shape == (4, 1, 257, 251)
assert batch_target.shape == (4, 1, 257, 251)

print("\nAll Dataset and DataLoader tests passed.")
