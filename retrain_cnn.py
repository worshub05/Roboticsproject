#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/home/shub_2605/ros2_ws/src/lane_follower_perception')
from lane_follower_perception.lane_cnn_model import LaneCNN

class LaneDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.image_files = sorted(self.data_dir.glob('img_*.png'))
        print(f"Found {len(self.image_files)} training images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        filename = img_path.name
        
        # Extract steering angle from filename: img_000001_0.851.png
        steering = float(filename.split('_')[-1].replace('.png', ''))
        
        # Load and preprocess image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # Resize to 160x100
        image = cv2.resize(image, (160, 100))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.FloatTensor(image), torch.FloatTensor([steering])

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load dataset
    data_dir = Path.home() / 'ros2_ws' / 'training_data'
    dataset = LaneDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = LaneCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"📊 Training samples: {len(dataset)}")
    print(f"🚀 Starting training for 40 epochs...\n")
    
    # Train
    for epoch in range(40):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/10 - Loss: {loss:.4f}")
    
    # Save model
    model_path = Path.home() / 'ros2_ws' / 'models' / 'lane_cnn_v2.pt'
    torch.save(model.state_dict(), str(model_path))
    print(f"\n✅ Retrained model saved to: {model_path}")
    print(f"   File size: {model_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()
