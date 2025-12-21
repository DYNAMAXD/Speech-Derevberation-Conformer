# Speech-Derevberation-Conformer
Conformer-based neural network for speech dereverberation using RT60 conditioning. Trained on 70k+ audio samples to remove room reverberation while preserving speech quality.

/*
# Speech Dereverberation Pipeline – README

## Overview

This repository implements a **log-Mel spectrogram–based speech dereverberation pipeline**.
The system is trained to map **reverberant log-Mel spectrograms** to **clean log-Mel spectrograms**,
and reconstructs waveform audio during inference using **HiFi-GAN**.

The workflow consists of:
- Dataset preprocessing and mapping
- Log-Mel feature extraction
- Model training on Mel-domain features
- Audio reconstruction during evaluation using a neural vocoder

---

## Checkpoint Usage

### Using a Checkpoint
- Checkpoints store **model weights only**
- Feature extraction parameters **must exactly match training**
- Inference assumes:
  - Sample rate: 16 kHz
  - Number of Mel bands: 80
  - FFT and hop settings consistent with training

Steps:
1. Load the model architecture used during training
2. Load checkpoint weights
3. Apply the same preprocessing pipeline
4. Perform inference in the log-Mel domain
5. Reconstruct waveform using HiFi-GAN

### Editing Configuration Before Loading a Checkpoint

Before loading a checkpoint, ensure the following parameters match training exactly:
- `sample_rate`
- `n_fft`
- `hop_length`
- `win_length`
- `n_mels`
- Mel filter configuration

Any mismatch will lead to incorrect feature alignment and degraded output quality.

---

## Dataset Preprocessing and File Mapping

### Dataset Flow

ARNI (Room Impulse Responses)
→ Raw Room Acoustic Signals
→ Convolution with Clean Speech
→ Reverberant Speech
→ Log-Mel Feature Extraction
→ Serialized Pickle Files

---

### Mapping Files

#### 1. ARNI → Raw Room Acoustic → Raw Speech

- Mapping file:
  `dataset/mapping/convolution_mapping_full.csv`

- Output stored in:
  `dataset/preProcessedSpeech/`

#### 2. Raw Room Acoustic → Raw Speech

- Mapping file:
  `dataset/preProcessedSpeech/preprocessed_mapping.csv`

- Output stored in:
  `dataset/finalRawSpeech/`

These CSV files define the correspondence between:
- Room impulse responses (ARNI)
- Clean speech files
- Generated reverberant speech samples

---

## Pickle File Structure

Each training sample is stored as a serialized `.pkl` file with the following structure:

-------------------------------------------------
conv_240-160593-0005_raw_...rt6.pkl
-------------------------------------------------
raw_features    : 80 × T  (Reverberant log-Mel)
clean_features  : 80 × T  (Clean log-Mel target)
rt60            : Reverberation time (float)
config          : Feature extraction parameters
-------------------------------------------------

- `raw_features`   : Reverberant log-Mel spectrogram
- `clean_features` : Target clean log-Mel spectrogram
- `rt60`           : Reverberation time of the room
- `config`         : Audio and feature extraction metadata

---

## Feature Extraction Pipeline

### Training Pipeline

Raw Audio (.wav)
→ STFT (512-point FFT)
→ Complex Spectrogram (257 bins)
→ Mel Filter Bank (257 → 80)
→ Log Compression
→ Log-Mel Spectrogram (80 bins)
→ Model
→ Clean Log-Mel Spectrogram (80 bins)

- **Input**  : Reverberant log-Mel (80 bins)
- **Output** : Clean log-Mel (80 bins)
- **Loss**   : Computed directly in the log-Mel domain

---

### Inference Pipeline

Raw Audio
→ STFT
→ Mel (257 → 80)
→ Log-Mel
→ Model
→ Clean Log-Mel
→ HiFi-GAN
→ Reconstructed Audio

---

## Audio Reconstruction and HiFi-GAN

### Why HiFi-GAN is Required

- Mel spectrograms are **not invertible**
- Standard inverse STFT cannot reconstruct audio from Mel features
- Direct Mel-to-waveform inversion is mathematically ill-posed

### Solution: HiFi-GAN

HiFi-GAN is used as a **neural vocoder** to synthesize waveforms from Mel spectrograms.

- Converts Mel spectrograms directly to waveform audio
- GAN-based architecture with generator and discriminator
- Used only during **evaluation and inference**

Reference:
https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/

---

## Training vs Evaluation Summary

### Training Phase
- Input  : Reverberant log-Mel (80 bins)
- Output : Clean log-Mel (80 bins)
- Loss  : Log-Mel domain loss
- Waveform reconstruction is **not performed**

### Evaluation Phase
- Input  : Raw audio
- Model operates fully in the log-Mel domain
- HiFi-GAN reconstructs waveform
- Output : Time-domain audio signal

---

## Key Notes

- The model **never operates on waveform audio**
- All learning occurs in the **log-Mel domain**
- HiFi-GAN is strictly for **post-processing**
- Feature extraction consistency is critical when using checkpoints
*/
