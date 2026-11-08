# Neural Audio Denoising

A speech enhancement system that combines **Digital Signal Processing (DSP)** and **Deep Learning** to suppress background noise in speech recordings.

The project explores the complete workflow of developing a neural-network-based audio denoising system, from audio preprocessing and spectral-mask estimation to model deployment and C++ inference.

## Project Overview

The objective is to develop a system that takes a noisy speech recording and produces an enhanced speech signal using a convolutional neural network (CNN).

The system operates in the time-frequency domain. Audio is transformed into a magnitude spectrogram using the Short-Time Fourier Transform (STFT). A neural network estimates a spectral mask, which is applied to the noisy signal to suppress noise while preserving speech characteristics.

The project combines Python-based model development with DSP processing and C++ inference.

## System Architecture

The denoising pipeline consists of five stages:

1. **Noisy Speech Input** – Load a noisy WAV recording.
2. **DSP Preprocessing** – Perform framing, STFT, magnitude-spectrum extraction and normalization.
3. **Neural Network Inference** – Train a CNN in PyTorch, export the model to ONNX and perform inference using C++.
4. **DSP Postprocessing** – Apply the estimated spectral mask, perform inverse STFT and overlap-add reconstruction.
5. **Enhanced Speech Output** – Generate the denoised WAV recording.

## Technical Approach

### 1. Audio Processing

- Load and process WAV audio recordings.
- Convert stereo audio to mono.
- Resample audio to a target sampling rate.
- Apply windowing and STFT to obtain time-frequency representations.
- Extract magnitude spectra and normalize the input features.
- Reconstruct audio using inverse STFT.

### 2. Dataset and Training Data Preparation

The system uses the **VoiceBank + DEMAND** dataset, containing paired clean and noisy speech recordings.

- Training set: 11,572 paired recordings.
- Test set: 824 paired recordings.
- Sampling rate: 16 kHz.
- Dataset license: CC BY 4.0.

The dataset is available from [Hugging Face – VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k).

Download the dataset:

```bash
pip install huggingface_hub

hf download JacobLinCool/VoiceBank-DEMAND-16k \
    --repo-type dataset \
    --local-dir data/dataset/parquet
```

Extract the audio recordings:

```bash
python -m python.extract_dataset
```

Validate the extracted dataset:

```bash
python -m python.validate_dataset
```

### 3. Neural Network

A convolutional neural network (CNN) is used to estimate a spectral mask from the normalized noisy magnitude spectrogram.

The model is developed and trained using PyTorch. The estimated mask is used to suppress noise in the time-frequency representation while retaining the phase information of the noisy input.

### 4. DSP Postprocessing

The estimated spectral mask is applied to the complex STFT representation of the noisy signal. The enhanced signal is then reconstructed using inverse STFT and overlap-add processing.

### 5. Model Deployment

The deployment workflow uses ONNX to export the trained PyTorch model and C++ for model inference. This provides a path from Python-based model development to a standalone inference implementation.

## Signal Processing Configuration

| Parameter | Configuration |
|---|---|
| Sampling rate | 16 kHz |
| FFT size | 512 |
| Hop length | 128 samples |
| Window | Hann |
| Training segment length | 2 seconds |
| STFT representation | Complex spectrum |
| Neural network input | Normalized magnitude spectrogram |
| Neural network output | Estimated spectral mask |

## Technologies

- **Programming:** Python, C++17
- **DSP:** NumPy, SciPy, Librosa, SoundFile
- **Deep Learning:** PyTorch, TorchAudio
- **Model Deployment:** ONNX, ONNX Runtime
- **Build System:** CMake
- **Visualization:** Matplotlib
- **Testing:** Pytest

## Installation

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

## Running the Project

Generate test audio:

```bash
python -m python.generate_test_audio
```

Run the STFT test:

```bash
python -m python.test_stft
```

Visualize the audio:

```bash
python -m python.visualize_audio
```

Run the CNN model test:

```bash
python -m python.models.test_model
```

## Key Learning Outcomes

This project provides practical experience in:

- Digital signal processing for speech enhancement.
- Time-frequency analysis using STFT and inverse STFT.
- Audio framing, windowing and overlap-add reconstruction.
- Spectral-mask estimation for noise suppression.
- Preparing paired audio datasets for supervised learning.
- Building and training convolutional neural networks using PyTorch.
- Evaluating speech enhancement performance.
- Deploying neural networks using ONNX.
- Integrating deep-learning inference into C++ applications.
- Developing and validating an end-to-end audio processing pipeline.

## Development Environment

- Ubuntu 22.04 LTS (WSL2)
- Python 3.10
- C++17
- CMake
