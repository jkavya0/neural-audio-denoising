
import torch

from python.models.denoising_model import DenoisingModel


def main():
    model = DenoisingModel()
    model.eval()

    # Example: one audio recording, 257 frequency bins,
    # and 626 time frames.
    test_input = torch.randn(1, 1, 257, 626)

    with torch.no_grad():
        predicted_mask = model(test_input)

    print("Input shape:", tuple(test_input.shape))
    print("Output shape:", tuple(predicted_mask.shape))
    print("Mask minimum:", predicted_mask.min().item())
    print("Mask maximum:", predicted_mask.max().item())

    assert predicted_mask.shape == test_input.shape
    assert torch.all(predicted_mask >= 0)
    assert torch.all(predicted_mask <= 1)

    print("All model forward-pass tests passed.")


if __name__ == "__main__":
    main()
