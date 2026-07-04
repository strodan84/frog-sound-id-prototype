#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID Prototype - Audio Preprocessing Pipeline
Converts raw 1D waveforms into fixed-length 2D Log-Mel Spectrograms.
"""

import os
import glob
import numpy as np
import librosa
import pandas as pd

MANIFEST_PATH = "data/dataset_manifest.csv"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Audio Hyperparameters for Bioacoustics
SAMPLE_RATE = 22050  # Standard audio sampling rate for ML
DURATION = 3         # Chunk duration in seconds
N_MELS = 128         # Number of Mel bands (height of image)
N_FFT = 1024         # Fast Fourier Transform window size
HOP_LENGTH = 512     # Stride length between frames

def preprocess_audio(file_path):
    """Loads, resamples, chunks/pads, and extracts Log-Mel Spectrogram."""
    try:
        # Load audio (automatically downsampled to single-channel 22.05kHz)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        target_len = SAMPLE_RATE * DURATION
        
        # Handle zero-padding for short clips, or slice long ones
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
            
        # Compute the Short-Time Fourier Transform onto a Mel Scale
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        
        # Convert power spectrogram to decibel units (logarithmic)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize between [0, 1] for stable neural network inputs
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        
        return log_mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    print("🎨 Transforming raw audio into Log-Mel Spectrogram features...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Cannot find manifest at {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    processed_records = []
    
    # Locate column names dynamically
    url_col = 'source_url' if 'source_url' in df.columns else df.columns[-1]
    species_col = 'species_code' if 'species_code' in df.columns else df.columns[0]
    id_col = 'xc_id' if 'xc_id' in df.columns else None

    for idx, row in df.iterrows():
        species_slug = str(row[species_col]).lower().replace(" ", "_")
        rec_id = row[id_col] if id_col and pd.notna(row[id_col]) else idx
        
        # Match file suffix dynamically (supporting .mp3 or .wav defaults)
        species_dir = os.path.join(RAW_DIR, species_slug)
        possible_files = glob.glob(os.path.join(species_dir, f"XC_{rec_id}.*"))
        
        if not possible_files:
            print(f"⚠️ Could not find raw file for asset {rec_id} in {species_dir}")
            continue
            
        raw_file_path = possible_files[0]
        
        # Run conversion
        spectrogram = preprocess_audio(raw_file_path)
        
        if spectrogram is not None:
            out_filename = f"XC_{rec_id}.npy"
            out_species_dir = os.path.join(PROCESSED_DIR, species_slug)
            os.makedirs(out_species_dir, exist_ok=True)
            
            out_path = os.path.join(out_species_dir, out_filename)
            np.save(out_path, spectrogram)
            
            # Save mapping details for the dataloader
            processed_records.append({
                "species_code": species_slug,
                "processed_path": out_path
            })
            
    # Save the processed array log
    pd.DataFrame(processed_records).to_csv("data/processed_manifest.csv", index=False)
    print(f"✅ Preprocessing completed. Spectrogram arrays saved to '{PROCESSED_DIR}/'")

if __name__ == "__main__":
    main()