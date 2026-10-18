#!/usr/bin/env python3
import numpy as np
import cv2
import os
from pathlib import Path

def generate_training_data(num_samples=2000):
    """Generate synthetic training data"""
    data_dir = Path.home() / 'ros2_ws' / 'training_data'
    data_dir.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} synthetic images...")
    
    for i in range(num_samples):
        # Random steering angle between -1.2 and 1.2 radians
        steering = np.random.uniform(-1.2, 1.2)
        
        # Create synthetic image (160x100 RGB)
        # Simulate lane following: brighter pixels toward the steering direction
        image = np.ones((100, 160, 3), dtype=np.uint8) * 50  # Dark background
        
        # Add lane line (white stripe in middle)
        center_offset = int(steering * 20)  # Steering affects lane position
        y_start, y_end = 30, 100
        x_center = 80 + center_offset
        cv2.line(image, (x_center-5, y_start), (x_center-10, y_end), (255, 255, 255), 3)
        cv2.line(image, (x_center+5, y_start), (x_center+10, y_end), (255, 255, 255), 3)
        
        # Add some noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Save with steering angle in filename
        filename = data_dir / f'img_{i+1:06d}_{steering:.3f}.png'
        cv2.imwrite(str(filename), image)
        
        if (i + 1) % 100 == 0:
            print(f"  ✅ Generated {i+1}/{num_samples} images")
    
    print(f"\n✅ Training data saved to: {data_dir}")
    print(f"   Total files: {len(list(data_dir.glob('*.png')))}")

if __name__ == '__main__':
    generate_training_data(num_samples=2000)
