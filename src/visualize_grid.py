#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID - Multi-Species Grid Visualizer
Generates a comparative grid of Mel-Spectrogram signatures for your README.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_MANIFEST = "data/processed_manifest.csv"

def generate_species_grid():
    if not os.path.exists(PROCESSED_MANIFEST):
        print("❌ Cannot find processed manifest. Run preprocess.py first.")
        return
        
    df = pd.read_csv(PROCESSED_MANIFEST)
    
    # Extract the first sample path for every unique species code
    species_representatives = df.groupby('species_code').first().reset_index()
    num_species = len(species_representatives)
    
    if num_species == 0:
        print("❌ No processed species arrays found in manifest.")
        return

    print(f"📊 Mapping a {num_species}-panel visual grid...")
    
    # Calculate grid layout dynamically (aiming for 2 or 3 columns)
    cols = 3 if num_species >= 3 else num_species
    rows = int(np.ceil(num_species / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True, sharey=True)
    
    # Flatten axes array for simple iteration if it's multi-dimensional
    if num_species > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for idx, row in species_representatives.iterrows():
        ax = axes[idx]
        species_slug = row['species_code']
        
        # Format Windows/Unix path strings cleanly
        file_path = row['processed_path'].replace('\\', '/')
        if file_path.startswith('../'):
            file_path = file_path[3:]
            
        try:
            # Load the 2D array matrix
            spectrogram = np.load(file_path)
            
            # Draw the heatmap panel
            im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
            
            # Title treatment formatting (e.g., "pseudacris_crucifer" -> "Pseudacris Crucifer")
            clean_title = species_slug.replace('_', ' ').title()
            ax.set_title(clean_title, fontsize=10, fontweight='bold', pad=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Load Error", ha='center', va='center', color='red')
            print(f"⚠️ Failed to load array for {species_slug}: {e}")
            
    # Hide any leftover empty panels if our grid math has remaining slots
    for extra_idx in range(num_species, len(axes)):
        fig.delaxes(axes[extra_idx])
        
    # Label layout boundaries
    fig.text(0.5, 0.01, 'Temporal Frames', ha='center', fontsize=11, fontweight='bold')
    fig.text(0.01, 0.5, 'Mel Frequency Bands', va='center', rotation='vertical', fontsize=11, fontweight='bold')
    
    os.makedirs("assets", exist_ok=True)
    output_path = "assets/species_spectrogram_grid.png"
    
    plt.tight_layout(rect=[0.02, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"🎨 Grid layout compiled successfully and saved to: {output_path}")

if __name__ == "__main__":
    generate_species_grid()