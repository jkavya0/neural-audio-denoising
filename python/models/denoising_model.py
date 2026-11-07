
import torch
import torch.nn as nn


class DenoisingModel(nn.Module):
    """
    CNN-based spectral mask estimator.

    Input:
        Normalized noisy magnitude spectrum
        Shape: (batch, 1, frequency_bins, time_frames)

    Output:
        Estimated spectral mask
        Shape: (batch, 1, frequency_bins, time_frames)
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
