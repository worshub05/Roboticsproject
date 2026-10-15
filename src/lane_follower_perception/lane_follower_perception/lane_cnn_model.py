"""
CNN Model for Lane Following
Standalone module (no ROS dependencies)
"""

import torch
import torch.nn as nn
import cv2
import numpy as np


class LaneCNN(nn.Module):
    """5-block CNN for steering prediction"""
    
    def __init__(self):
        super(LaneCNN, self).__init__()
        
        # Convolutional blocks with BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global average pooling and fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        """Forward pass: image -> steering angle"""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def preprocess_image(image):
    """
    Convert camera image to model input tensor
    
    Args:
        image: numpy array, shape (H, W, 3), dtype uint8
    
    Returns:
        torch.Tensor, shape (1, 3, 100, 160)
    """
    # Crop top 25% (remove sky/irrelevant features)
    if image.shape[0] > 100:
        image = image[int(image.shape[0] * 0.25):, :, :]
    
    # Resize to standard size
    image = cv2.resize(image, (160, 100))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert HWC to CHW (PyTorch format)
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    return torch.from_numpy(image).unsqueeze(0)


def load_model(checkpoint_path, device='cpu'):
    """
    Load trained CNN model from checkpoint
    
    Args:
        checkpoint_path: path to .pt file
        device: 'cpu' or 'cuda'
    
    Returns:
        LaneCNN model in eval mode
    """
    model = LaneCNN().to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"[MODEL] Loaded: {model.count_parameters()} parameters")
        print(f"[MODEL] From: {checkpoint_path}")
        return model
    except FileNotFoundError:
        print(f"[MODEL] Warning: File not found: {checkpoint_path}")
        print(f"[MODEL] Using untrained model (random initialization)")
        model.eval()
        return model


def infer_steering(model, image, device='cpu'):
    """
    Run inference: image -> steering angle
    
    Args:
        model: LaneCNN model
        image: numpy array from camera
        device: 'cpu' or 'cuda'
    
    Returns:
        float: steering angle in [-1.2, 1.2] radians
    """
    with torch.no_grad():
        # Preprocess
        tensor = preprocess_image(image)
        tensor = tensor.to(device)
        
        # Inference
        steering = model(tensor).item()
        
        # Clamp to valid range
        steering = max(-1.2, min(1.2, float(steering)))
    
    return steering
