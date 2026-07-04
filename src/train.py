#!/usr/bin/env python3
"""
🌿 iNaturalist Sound ID Prototype - Training Framework
Loads balanced data batches, sets up a vision backbone classifier,
and processes optimization runs with evaluation handling.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models

# Import our custom structural dataset hooks
from dataset import BioacousticDataset, get_spec_augment

PROCESSED_MANIFEST = "data/processed_manifest.csv"

def train_prototype_model():
    if not os.path.exists(PROCESSED_MANIFEST):
        print(f"❌ Error: Cannot find your processed data map at {PROCESSED_MANIFEST}")
        print("Please verify that 'python src/preprocess.py' completed successfully.")
        return

    print("🦅 Initializing Bioacoustic Neural Architecture...")
    
    # 1. Instantiate dataset and structural class balancing samplers
    train_dataset = BioacousticDataset(PROCESSED_MANIFEST, transform=get_spec_augment())
    num_classes = len(train_dataset.unique_species)
    
    sampler_weights = train_dataset.get_sampler_weights()
    sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(sampler_weights), replacement=True)
    
    # Production Dataloader setup
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    
    print(f"🔬 Found {len(train_dataset)} processed arrays across {num_classes} frog target taxa.")

    # 2. Setup standard high-performance vision backbone (ResNet-18)
    # We modify the very first layer because spectrograms are single-channel grayscale (1), not RGB (3)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Adapt final fully-connected output layer to match our exact frog species count
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Define Loss and Optimization functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # 3. Model Training Loop (Prototype compilation test)
    print("\n🏃 Running training iteration to compile execution graph...")
    model.train()
    
    for epoch in range(1, 3):  # Run 2 short iterations to demonstrate the graph works
        running_loss = 0.0
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"   Epoch [{epoch}/2] -> Normalized Loss Matrix: {running_loss / len(train_loader):.4f}")
        
    # 4. Serialize model for edge constraints (TorchScript production bundle)
    print("\n📱 Compiling architecture into serialized production asset...")
    model.eval()
    example_input = torch.rand(1, 1, 128, 130)  # Standard dimensions of our Mel-spectrogram
    traced_script_module = torch.jit.trace(model, example_input)
    
    os.makedirs("models", exist_ok=True)
    export_path = "models/sound_id_frog_resnet18.pt"
    traced_script_module.save(export_path)
    
    print(f"🎉 Edge asset compiled successfully and exported to: {export_path}")

if __name__ == "__main__":
    train_prototype_model()