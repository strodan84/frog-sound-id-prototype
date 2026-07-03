# 🌿 iNaturalist Sound ID: End-to-End Bioacoustic Pipeline Prototype 

An applied ML engineering showcase demonstrating an end-to-end signal processing and computer vision pipeline for classifying animal vocalizations. This architecture transforms raw 1D audio waveforms into 2D Mel-spectrogram images, handles heavy class imbalances via custom batch sampling, trains a convolutional neural network (CNN), and optimizes the final asset via INT8 quantization for ultra-low latency mobile deployment.

## 🏗️ Pipeline Architecture

```mermaid
graph TD
    A[Raw 1D Audio .wav file] -->|22.05kHz Resampling| B[STFT Signal Processing]
    B -->|Hann Windowing & Mel-Scale| C[2D Mel-Spectrogram Decibels]
    C -->|Weighted Random Sampler| D[PyTorch EfficientNet / ResNet]
    D -->|Frequency & Time Masking| E[Species Inference Matrix]
    E -->|Post-Training Quantization| F[Edge-Optimized TFLite Engine]
