#!/usr/bin/env python3
"""
iNaturalist Sound ID Prototype - Data Ingestion Engine
Downloads high-quality amphibian recordings from the Xeno-Canto REST API 
for key indicator species native to the Southern New Jersey Coastal Plain.
"""

import os
import time
import requests
import pandas as pd

# Define target indicator species (Scientific Name -> Common Name)
TARGET_SPECIES = {
    "Dryophytes andersonii": "Pine Barrens Treefrog",
    "Anaxyrus fowleri": "Fowlers Toad",
    "Lithobates clamitans": "Green Frog",
    "Lithobates catesbeianus": "Bullfrog",
    "Pseudacris kalmi": "New Jersey Chorus Frog",
    "Pseudacris crucifer": "Spring Peeper",
    "Lithobates sphenocephalus": "Southern Leopard Frog",
    "Lithobates sylvaticus": "Wood Frog",
    "Hyla versicolor": "Gray Treefrog",
    "Hyla chrysoscelis": "Copes Gray Treefrog"
}

DATA_DIR = "data/raw"
MANIFEST_PATH = "data/dataset_manifest.csv"
MAX_RECORDINGS_PER_SPECIES = 5  # Keeps the initial prototype pipeline fast

def setup_directories():
    """Ensure raw audio target folders exist."""
    for species in TARGET_SPECIES.keys():
        folder_name = species.lower().replace(" ", "_")
        os.makedirs(os.path.join(DATA_DIR, folder_name), exist_ok=True)

def fetch_recordings_metadata(species_name):
    """Query the Xeno-Canto API for high-quality audio files from the US."""
    base_url = "https://xeno-canto.org/api/2/recordings"
    
    # Construct an API query string restricting group, location, and quality
    # grp:frogs -> targets amphibians instead of birds
    query = f'{species_name} cnt:"United States" grp:frogs q:A'
    params = {"query": query}
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Fallback to Quality B if Quality A returns zero results
        if int(data.get("numRecordings", 0)) == 0:
            query = f'{species_name} cnt:"United States" grp:frogs q:B'
            response = requests.get(base_url, params={"query": query}, timeout=15)
            data = response.json()
            
        return data.get("recordings", [])
    except Exception as e:
        print(f"⚠️ Error querying API for {species_name}: {e}")
        return []

def download_audio_file(download_url, output_path):
    """Stream download the binary asset with proper connection error handling."""
    try:
        with requests.get(download_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"   ❌ Download failed for {download_url}: {e}")
        return False

def main():
    print("Starting Bioacoustic Data Ingestion Pipeline...")
    setup_directories()
    
    manifest_records = []
    
    for scientific_name, common_name in TARGET_SPECIES.items():
        print(f"\n🔍 Processing: {common_name} ({scientific_name})")
        recordings = fetch_recordings_metadata(scientific_name)
        
        if not recordings:
            print(f"   No high-quality recordings found via API filters.")
            continue
            
        # Slice to ensure balanced class distributions for the prototype
        selected_recordings = recordings[:MAX_RECORDINGS_PER_SPECIES]
        print(f"   Found {len(recordings)} matches. Downloading top {len(selected_recordings)}...")
        
        species_slug = scientific_name.lower().replace(" ", "_")
        
        for rec in selected_recordings:
            rec_id = rec.get("id")
            # The API returns direct audio streaming paths at: xeno-canto.org/{id}/download
            download_url = f"https://xeno-canto.org/{rec_id}/download"
            
            # Xeno-canto files are typically delivered as MP3s
            filename = f"XC_{rec_id}.mp3"
            relative_path = os.path.join(DATA_DIR, species_slug, filename)
            
            print(f"   📥 Fetching asset ID {rec_id}...", end="", flush=True)
            
            if os.path.exists(relative_path):
                print(" [Skipped: Already Exists]")
                success = True
            else:
                success = download_audio_file(download_url, relative_path)
                if success:
                    print(" [Done]")
                # Polite rate-limiting to preserve server bandwidth
                time.sleep(1.5)
                
            if success:
                manifest_records.append({
                    "species_code": species_slug,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                    "xc_id": rec_id,
                    "local_path": relative_path,
                    "quality": rec.get("q"),
                    "length_sec": rec.get("length")
                })
                
    # Export a structured manifest file used by our PyTorch Dataset class
    if manifest_records:
        df = pd.DataFrame(manifest_records)
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"\n✅ Pipeline Complete! Metadata manifest exported to {MANIFEST_PATH}")
    else:
        print("\n❌ Pipeline completed with 0 downloaded records.")

if __name__ == "__main__":
    main()