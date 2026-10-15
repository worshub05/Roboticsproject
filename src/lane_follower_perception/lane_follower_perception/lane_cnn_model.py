"""
CNN Model for Lane Following
5-block convolutional neural network with 553,021 parameters
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

class LaneCNN(nn.Module):
    """
    5-block CNN for steering angle prediction
    Input: 160x100x3 RGB image
    Output: Steering angle [-1.2, 1.2] radians
    """

    def __init__(self):
        super(LaneCNN, self).__init__()

        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Block 4: 128 -> 256 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Block 5: 256 -> 512 channels
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass"""
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout(x)

        # Block 5
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Clamp steering angle to [-1.2, 1.2]
        x = torch.clamp(x, -1.2, 1.2)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def preprocess_image(image_array, target_size=(160, 100)):
    """
    Preprocess image for CNN inference

    Args:
        image_array: numpy array (H, W, C) in range [0, 255]
        target_size: tuple (width, height)

    Returns:
        torch tensor (C, H, W) normalized to [0, 1]
    """
    # Crop top 25% (road ahead is more relevant)
    crop_height = int(image_array.shape[0] * 0.75)
    image_cropped = image_array[int(image_array.shape[0] * 0.25):, :, :]

    # Resize to target size
    image_pil = Image.fromarray(image_cropped.astype(np.uint8))
    image_resized = image_pil.resize(target_size, Image.LANCZOS)

    # Convert to numpy and normalize to [0, 1]
    image_np = np.array(image_resized).astype(np.float32) / 255.0

    # Convert HWC to CHW
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor


def load_model(model_path, device='cpu'):
    """
    Load trained model from checkpoint

    Args:
        model_path: path to .pt file
        device: 'cpu' or 'cuda'

    Returns:
        model: LaneCNN instance with loaded weights
    """
    model = LaneCNN().to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

    return model


def infer_steering(model, image_array, device='cpu'):
    """
    Infer steering angle from image

    Args:
        model: LaneCNN instance
        image_array: numpy array (H, W, 3) in range [0, 255]
        device: 'cpu' or 'cuda'

    Returns:
        steering_angle: float in [-1.2, 1.2] radians
    """
    with torch.no_grad():
        # Preprocess image
        image_tensor = preprocess_image(image_array)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Forward pass
        steering = model(image_tensor)
        steering_angle = steering.cpu().item()

    return steering_angle
