# Neural Audio Denoising

A speech enhancement project exploring digital signal processing, deep learning, and cross-platform model deployment using Python, PyTorch, ONNX Runtime, and C++.

## Project Objective

Develop an audio denoising application that uses a deep neural network to suppress background noise while preserving speech quality.

The pipeline is:

1. Audio input and preprocessing
2. STFT-based time-frequency analysis
3. Neural network training using PyTorch
4. Spectral mask estimation and noise suppression
5. ONNX model export
6. C++ inference using ONNX Runtime
7. Performance evaluation and optimizationi thing&#x20;

## Technologies

- Python
- C++17
- NumPy and SciPy
- Librosa and SoundFile
- PyTorch and TorchAudio
- ONNX and ONNX Runtime
- CMake
- Matplotlib
- Pytest

**Initial signal configuration:**

| Parameter       |     Value |
| --------------- | --------: |
| Sampling rate   |    16 kHz |
| Signal duration | 3 seconds |
| FFT size        |       512 |
| Hop length      |       128 |
| Window          |      Hann |

A synthetic test signal containing 440 Hz and 880 Hz sinusoidal components was generated to verify the audio-processing pipeline.

**STFT reconstruction verification:**

- Original samples: 48,000
- Reconstructed samples: 48,000
- Reconstruction MSE: approximately `9.92e-16`

### Visualization Results

#### Clean Audio Waveform



#### Clean Audio Spectrogram


## Repository Structure

```text
neural-audio-denoising/
├── cpp/
│   ├── include/
│   └── src/
├── data/
│   ├── clean/
│   └── noisy/
├── docs/
├── models/
├── python/
│   ├── preprocessing/
│   │   ├── audio_io.py
│   │   └── stft.py
│   ├── generate_test_audio.py
│   ├── test_stft.py
│   └── visualize_audio.py
├── results/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/jkavya0/neural-audio-denoising.git
cd neural-audio-denoising
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic test audio:

```bash
python -m python.generate_test_audio
```

Run the STFT reconstruction test:

```bash
python -m python.test_stft
```

Generate waveform and spectrogram visualizations:

```bash
python -m python.visualize_audio
```

## Development Environment

- Ubuntu 22.04
- Python 3.10 or compatible version
- G++ 11.4 or compatible C++17 compiler
- CMake 3.22 or compatible version

---

