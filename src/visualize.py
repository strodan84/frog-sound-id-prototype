#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID - Feature Visualization Engine
Generates and saves a high-density Mel-Spectrogram plot for your portfolio.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_MANIFEST = "data/processed_manifest.csv"

def generate_spectrogram_plot():
    if not os.path.exists(PROCESSED_MANIFEST):
        print("❌ Cannot find processed manifest. Run preprocess.py first.")
        return
        
    df = pd.read_csv(PROCESSED_MANIFEST)
    # Grab the very first available processed file to visualize
    row = df.iloc[0]
    file_path = row['processed_path'].replace('\\', '/')
    if file_path.startswith('../'):
        file_path = file_path[3:]
        
    species_name = row['species_code']
    
    print(f"📊 Loading spectrogram matrix for: {species_name}")
    spectrogram = np.load(file_path)
    
    # Plot the matrix
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    
    plt.title(f"Log-Mel Spectrogram Signature: {species_name.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
    plt.ylabel("Mel Frequency Bands", fontsize=10)
    plt.xlabel("Temporal Frames", fontsize=10)
    plt.colorbar(format='%+2.0f dB')
    
    # Save it out as an asset
    os.makedirs("assets", exist_ok=True)
    output_path = "assets/sample_spectrogram.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"🎨 Visual asset generated successfully and saved to: {output_path}")

if __name__ == "__main__":
    generate_spectrogram_plot()