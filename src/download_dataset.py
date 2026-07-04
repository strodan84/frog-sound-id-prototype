#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID Prototype - Direct Asset Downloader
Reads the existing dataset_manifest.csv file and directly streams 
the audio assets using the URLs provided.
"""

import os
import time
import requests
import pandas as pd

MANIFEST_PATH = "data/dataset_manifest.csv"
DATA_DIR = "data/raw"

def download_audio_file(download_url, output_path):
    """Stream download the binary asset safely from a direct URL."""
    # Fix any accidental double 'o' typos in the URL string dynamically
    if "xeno-cantoo.org" in download_url:
        download_url = download_url.replace("xeno-cantoo.org", "xeno-canto.org")
        
    try:
        # Xeno-canto requires a user-agent string so it doesn't block the request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        with requests.get(download_url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed for {download_url}: {e}")
        return False

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"❌ Error: Cannot find your manifest file at {MANIFEST_PATH}")
        print("Please ensure your CSV is saved there before running.")
        return

    print("🚀 Reading your custom manifest and starting direct downloads...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Clean up column names just in case there are trailing spaces
    df.columns = df.columns.str.strip()
    
    # Dynamically find your URL column (assuming it's the last column or named 'source_url')
    url_col = 'source_url' if 'source_url' in df.columns else df.columns[-1]
    species_col = 'species_code' if 'species_code' in df.columns else df.columns[0]
    id_col = 'xc_id' if 'xc_id' in df.columns else None

    successful_downloads = 0

    for idx, row in df.iterrows():
        species_slug = str(row[species_col]).lower().replace(" ", "_")
        download_url = str(row[url_col]).strip()
        
        # Append /download to xeno-canto page URLs if not already present
        if "xeno-canto.org" in download_url and not download_url.endswith("/download"):
            # If the URL is just a base link like https://xeno-canto.org/12345
            if download_url.split('/')[-1].isdigit():
                rec_id = download_url.split('/')[-1]
                download_url = f"https://xeno-canto.org/{rec_id}/download"
        
        # Determine file name
        rec_id = row[id_col] if id_col and pd.notna(row[id_col]) else idx
        filename = f"XC_{rec_id}.mp3"
        
        # Build paths
        species_dir = os.path.join(DATA_DIR, species_slug)
        os.makedirs(species_dir, exist_ok=True)
        relative_path = os.path.join(species_dir, filename)
        
        print(f"📥 [{idx+1}/{len(df)}] Fetching asset for {species_slug}...", end="", flush=True)
        
        if os.path.exists(relative_path):
            print(" [Skipped: Exists]")
            successful_downloads += 1
        else:
            # If your URL is just a reference link, we convert it to the download stream
            success = download_audio_file(download_url, relative_path)
            if success:
                print(" [Done]")
                successful_downloads += 1
            time.sleep(1.2) # Polite padding between server requests

    print(f"\n✅ Pipeline Complete! Successfully verified/downloaded {successful_downloads} assets.")

if __name__ == "__main__":
    main()







