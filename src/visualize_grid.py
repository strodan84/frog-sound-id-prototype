#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID - Multi-Species Grid Visualizer (Common Names)
Generates a comparative grid of Mel-Spectrogram signatures labeled with common names.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RAW_MANIFEST = "data/dataset_manifest.csv"
PROCESSED_MANIFEST = "data/processed_manifest.csv"

def generate_species_grid():
    if not os.path.exists(PROCESSED_MANIFEST) or not os.path.exists(RAW_MANIFEST):
        print("❌ Cannot find manifest files. Ensure pipelines have run successfully.")
        return
        
    # Load data maps
    df_raw = pd.read_csv(RAW_MANIFEST)
    df_proc = pd.read_csv(PROCESSED_MANIFEST)
    
    # Clean string buffers to ensure flawless merging
    df_raw.columns = df_raw.columns.str.strip()
    df_proc.columns = df_proc.columns.str.strip()
    
    # Identify key tracking columns dynamically
    raw_species_col = 'species_code' if 'species_code' in df_raw.columns else df_raw.columns[0]
    common_name_col = 'common_name' if 'common_name' in df_raw.columns else None
    
    if common_name_col is None:
        print("⚠️ 'common_name' column missing from raw manifest. Defaulting to short names.")
        name_mapping = {}
    else:
        # Create a fast lookup table: { 'pseudacris_crucifer': 'Spring Peeper' }
        name_mapping = dict(zip(df_raw[raw_species_col].str.lower(), df_raw[common_name_col]))

    # Grab the first processed sample path for every unique species code
    species_representatives = df_proc.groupby('species_code').first().reset_index()
    num_species = len(species_representatives)
    
    if num_species == 0:
        print("❌ No processed species arrays found in manifest.")
        return

    print(f"📊 Mapping a {num_species}-panel visual grid with common names...")
    
    # Calculate grid layout dimensions dynamically
    cols = 3 if num_species >= 3 else num_species
    rows = int(np.ceil(num_species / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True, sharey=True)
    
    if num_species > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for idx, row in species_representatives.iterrows():
        ax = axes[idx]
        species_slug = row['species_code'].lower()
        
        # Check our lookup dictionary for the clean common name, fall back to capitalized slug
        display_label = name_mapping.get(species_slug, species_slug.replace('_', ' ').title())
        
        # Format Windows/Unix path strings cleanly
        file_path = row['processed_path'].replace('\\', '/')
        if file_path.startswith('../'):
            file_path = file_path[3:]
            
        try:
            spectrogram = np.load(file_path)
            
            # Use the 'magma' or 'viridis' colormap for clean bioacoustic rendering
            ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
            ax.set_title(display_label, fontsize=11, fontweight='bold', pad=8)
            
        except Exception as e:
            ax.text(0.5, 0.5, "Load Error", ha='center', va='center', color='red')
            print(f"⚠️ Failed to load array for {species_slug}: {e}")
            
    # Remove empty subplots from the frame layout if grid is partially filled
    for extra_idx in range(num_species, len(axes)):
        fig.delaxes(axes[extra_idx])
        
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