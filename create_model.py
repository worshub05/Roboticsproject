#!/usr/bin/env python3
"""
Initialize and save a fresh LaneCNN model
"""
import torch
import sys
from pathlib import Path

# Add package to path
ros2_ws = Path.home() / 'ros2_ws'
sys.path.insert(0, str(ros2_ws / 'src' / 'lane_follower_perception'))

from lane_follower_perception.lane_cnn_model import LaneCNN

def main():
    print("Creating new LaneCNN model...")

    # Create model
    model = LaneCNN()
    param_count = model.count_parameters()
    print(f"✓ Model created with {param_count} parameters")

    # Save model
    models_dir = ros2_ws / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / 'lane_cnn.pt'
    torch.save(model.state_dict(), model_path)

    file_size = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model saved to: {model_path}")
    print(f"  File size: {file_size:.2f} MB")

    print("\nModel architecture:")
    print(model)

if __name__ == '__main__':
    main()
