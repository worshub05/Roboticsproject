#!/usr/bin/env python3
"""
Generate synthetic training data for CNN model
Creates images with lane visualization and steering angle labels
"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random

def generate_synthetic_image(steering_angle, width=160, height=100, noise_std=10):
    """
    Generate synthetic lane-following image

    Args:
        steering_angle: steering angle in radians [-1.2, 1.2]
        width: image width
        height: image height
        noise_std: gaussian noise standard deviation

    Returns:
        numpy array (H, W, 3) with road and lane
    """
    # Create base road (dark gray)
    image = Image.new('RGB', (width, height), color=(50, 50, 50))
    draw = ImageDraw.Draw(image)

    # Lane center offset based on steering angle
    # steering of 0 = center lane
    # positive steering = turn right
    # negative steering = turn left
    center_x = width / 2
    lane_offset = (steering_angle / 1.2) * (width / 4)  # Scale offset

    # Draw lane lines (bright yellow)
    lane_width = 30

    # Left lane line
    left_x = center_x - lane_width/2 + lane_offset
    draw.line([(left_x, 0), (left_x + 10, height)], fill=(255, 255, 0), width=3)

    # Right lane line
    right_x = center_x + lane_width/2 + lane_offset
    draw.line([(right_x, 0), (right_x + 10, height)], fill=(255, 255, 0), width=3)

    # Center lane marking (dashed)
    for y in range(0, height, 20):
        x = center_x + lane_offset
        draw.line([(x, y), (x, min(y+10, height))], fill=(200, 200, 0), width=2)

    # Road markings ahead (to indicate motion)
    road_color = (100, 100, 100)
    for y_base in range(0, height, 15):
        y = y_base
        draw.line([(0, y), (width, y)], fill=road_color, width=1)

    # Convert to numpy and add noise
    image_np = np.array(image).astype(np.float32)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, image_np.shape)
    image_np = np.clip(image_np + noise, 0, 255)

    return image_np.astype(np.uint8)


def main():
    print("Generating synthetic training data...")

    # Paths
    ros2_ws = Path.home() / 'ros2_ws'
    training_dir = ros2_ws / 'training_data'
    training_dir.mkdir(parents=True, exist_ok=True)

    # Generate images
    num_images = 500
    for i in range(num_images):
        # Random steering angle
        steering_angle = random.uniform(-1.2, 1.2)

        # Generate image
        image = generate_synthetic_image(steering_angle)

        # Save with steering angle in filename
        filename = f"img_{i+1:06d}_{steering_angle:.3f}.png"
        filepath = training_dir / filename

        Image.fromarray(image).save(filepath)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{num_images} images...")

    print(f"\n✓ Generated {num_images} images")
    print(f"✓ Saved to: {training_dir}")
    print(f"  Total size: {sum(f.stat().st_size for f in training_dir.glob('*.png')) / (1024*1024):.1f} MB")


if __name__ == '__main__':
    main()
