#!/usr/bin/env python3
"""
Retrain CNN model on synthetic training data
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import re

# Add package to path
ros2_ws = Path.home() / 'ros2_ws'
sys.path.insert(0, str(ros2_ws / 'src' / 'lane_follower_perception'))

from lane_follower_perception.lane_cnn_model import LaneCNN


class SyntheticLaneDataset(Dataset):
    """Dataset for synthetic lane images"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images = sorted(self.data_dir.glob('*.png'))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load image
        image = np.array(Image.open(img_path)).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Extract steering angle from filename
        # Format: img_XXXXXX_Y.YYY.png
        match = re.search(r'img_\d+_([-\d.]+)\.png', img_path.name)
        if match:
            steering = float(match.group(1))
        else:
            steering = 0.0

        return image, torch.tensor([steering], dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    print("Retraining CNN model...")

    # Paths
    training_dir = ros2_ws / 'training_data'
    models_dir = ros2_ws / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    if not training_dir.exists():
        print("Error: training_data directory not found!")
        print(f"Please run generate_training_data.py first")
        return

    # Check for training data
    images = list(training_dir.glob('*.png'))
    if not images:
        print("Error: No training images found!")
        print(f"Please run generate_training_data.py first")
        return

    print(f"Found {len(images)} training images")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = SyntheticLaneDataset(training_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = LaneCNN().to(device)
    param_count = model.count_parameters()
    print(f"Model: {param_count} parameters")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 50)

    for epoch in range(num_epochs):
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss:.4f}")

    # Save model
    model_path = models_dir / 'lane_cnn_retrained.pt'
    torch.save(model.state_dict(), model_path)

    file_size = model_path.stat().st_size / (1024 * 1024)
    print("-" * 50)
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {model_path}")
    print(f"  File size: {file_size:.2f} MB")


if __name__ == '__main__':
    main()
