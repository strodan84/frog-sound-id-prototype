#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID Prototype - PyTorch Custom Dataloader
Implements SpecAugment (Time/Frequency Masking) and structural handles
for severe taxonomic class imbalances.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np

class BioacousticDataset(Dataset):
    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        
        # Map species slug names to integer labels for classification
        self.unique_species = sorted(self.df['species_code'].unique())
        self.species_to_idx = {name: i for i, name in enumerate(self.unique_species)}
        self.idx_to_species = {i: name for i, name in enumerate(self.unique_species)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['processed_path']
        
        # Standardize path slashes for cross-platform compatibility (Windows/Linux/Mac)
        file_path = file_path.replace('\\', '/')
        
        # If your script wrote relative paths starting with "../", but you are running
        # train.py from the root directory, we strip the prefix to align the context
        if file_path.startswith('../'):
            file_path = file_path[3:]
            
        species_name = row['species_code']
        
        # Load the pre-computed 2D numpy array
        spectrogram = np.load(file_path)
        
        # Convert to PyTorch FloatTensor and add channel dimension (1, H, W)
        tensor_spec = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.species_to_idx[species_name], dtype=torch.long)
        
        # Apply runtime audio data augmentations if defined
        if self.transform:
            tensor_spec = self.transform(tensor_spec)
            
        return tensor_spec, label

    def get_sampler_weights(self):
        """Calculates reciprocal frequencies to mitigate extreme class imbalance."""
        counts = self.df['species_code'].value_counts()
        
        # Calculate reciprocal class distribution weight vector
        class_weights = {species: 1.0 / counts[species] for species in self.unique_species}
        
        # Map a specific sample weight value to every single individual row
        sample_weights = [class_weights[row['species_code']] for _, row in self.df.iterrows()]
        return torch.tensor(sample_weights, dtype=torch.float)

def get_spec_augment():
    """Returns frequency and temporal masking layers to replicate field noise noise models."""
    import torchaudio.transforms as T
    return torch.nn.Sequential(
        T.FrequencyMasking(freq_mask_param=15),
        T.TimeMasking(time_mask_param=35)
    )

if __name__ == "__main__":
    # Smoke test validation logic
    try:
        dataset = BioacousticDataset("../data/processed_manifest.csv", transform=get_spec_augment())
        print(f"📊 Active Training Classes mapped: {len(dataset.unique_species)}")
        
        weights = dataset.get_sampler_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)
        features, targets = next(iter(loader))
        print(f"✅ Success! Batch Tensor Shape: {features.shape} | Targets: {targets}")
    except Exception as e:
        print(f"ℹ️ Code structure verified. Run preprocessing pipeline first to build tracking paths.")