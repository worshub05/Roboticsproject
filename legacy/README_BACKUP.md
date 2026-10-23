# CNN Lane Following System for Warehouse Robots

A complete autonomous lane-following system for warehouse robot navigation using Convolutional Neural Networks (CNN), ROS 2 Humble, and Gazebo simulation.

## Project Overview

This project implements an end-to-end CNN-based perception and control system for warehouse robots:
- **CNN Model**: 5-block convolutional neural network with 553,021 parameters
- **ROS 2 Integration**: Humble distribution with proper node architecture
- **Gazebo Simulation**: Warehouse environment with sensors (camera, LiDAR, IMU)
- **Training Pipeline**: Synthetic data generation and model retraining

## System Architecture

```
Camera → Preprocessing → CNN Model → Steering Prediction
                           ↓
                    Differential Drive Controller
                           ↓
                      Robot Motion (ROS 2)
                           ↓
                      Gazebo Simulation
```

## Features

✅ CNN-based steering prediction ([-1.2, 1.2] radians)  
✅ ROS 2 Humble node architecture  
✅ Gazebo 11 physics simulation  
✅ Sensor integration (camera 30Hz, LiDAR 10Hz, IMU 50Hz)  
✅ Differential drive kinematics  
✅ Synthetic training data generation  
✅ Model training and retraining pipeline  
✅ Warehouse environment with obstacles  

## Directory Structure

```
cnn-lane-following/
├── src/
│   └── lane_follower_perception/
│       ├── lane_follower_perception/
│       │   ├── __init__.py
│       │   ├── lane_cnn_model.py          # CNN model class
│       │   └── cnn_inference_node.py      # ROS 2 inference node
│       ├── launch/
│       │   └── warehouse.launch.py        # Gazebo + nodes launch
│       └── setup.py
├── worlds/
│   └── warehouse.world                    # Gazebo SDF world
├── models/
│   └── lane_cnn_retrained.pt              # Trained model weights
├── training_data/
│   └── (500 synthetic training images)
├── create_model.py                        # Initialize model
├── generate_training_data.py              # Synthetic data generator
├── retrain_cnn.py                         # Training pipeline
├── differential_drive_controller.py       # Steering → wheel commands
├── debug_movement.py                      # Movement debugger
└── test_manual_movement.py                # Manual movement test
```

## Installation & Setup

### Prerequisites
- Ubuntu 22.04 (or WSL on Windows)
- ROS 2 Humble
- Gazebo 11
- Python 3.10+
- PyTorch

### Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/cnn-lane-following.git
cd cnn-lane-following
colcon build

# Source the workspace
source install/setup.bash

# Create initial model
python3 create_model.py

# Generate training data
python3 generate_training_data.py

# Train the model
python3 retrain_cnn.py

# Launch the system
ros2 launch lane_follower_perception warehouse.launch.py
```

## CNN Model Architecture

```
Input: 160×100×3 (RGB image)
    ↓
Conv Block 1: 32 filters, 3×3 kernel, BatchNorm
    ↓
Conv Block 2: 64 filters, BatchNorm
    ↓
Conv Block 3: 128 filters, BatchNorm
    ↓
Conv Block 4: 256 filters, BatchNorm
    ↓
Conv Block 5: 512 filters, BatchNorm + Adaptive AvgPool
    ↓
FC Layer 1: 512 → 256
    ↓
FC Layer 2: 256 → 1
    ↓
Output: Steering Angle [-1.2, 1.2] radians
```

**Total Parameters**: 553,021  
**Training Loss**: 0.5971 → 0.0132 (10 epochs)

## Key Components

### 1. CNN Inference Node (`cnn_inference_node.py`)
- Subscribes to `/camera/image_raw` at 30 Hz
- Publishes steering predictions to `/cmd_vel`
- Automatically uses GPU if available
- Configurable inference rate and linear speed

### 2. Differential Drive Controller (`differential_drive_controller.py`)
- Converts steering angle to differential wheel velocities
- Wheel separation: 0.3m
- Subscribes to `/cmd_vel`, publishes to `/robot/cmd_vel`
- Implements kinematic equations for smooth motion

### 3. Gazebo Environment
- 100m × 100m warehouse space
- Warehouse shelving with 12 shelves in 3 aisles
- Dynamic human obstacles
- Physics: ODE engine, 0.001s timestep

### 4. Sensor Suite
- **Camera**: 30 Hz, 640×480, RGB
- **LiDAR**: 10 Hz, 360° scan
- **IMU**: 50 Hz, 6-axis
- **Odometry**: 30 Hz position/velocity feedback

## Training & Data

### Synthetic Data Generation
```bash
python3 generate_training_data.py
```
Generates 500 training images with:
- Procedural lane visualization
- Gaussian noise (σ=10)
- Steering angles: [-1.2, 1.2] radians

### Model Training
```bash
python3 retrain_cnn.py
```
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: MSELoss
- Epochs: 10

## Debugging

### Monitor Movement
```bash
python3 debug_movement.py
```
Monitors `/cmd_vel`, `/robot/cmd_vel`, and `/robot/odom` topics.

### Manual Movement Test
```bash
python3 test_manual_movement.py
```
Sends direct commands to robot to verify differential drive plugin response.

## ROS 2 Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | IN | Camera images at 30 Hz |
| `/cmd_vel` | geometry_msgs/Twist | OUT | CNN steering predictions |
| `/robot/cmd_vel` | geometry_msgs/Twist | IN | Wheel velocity commands |
| `/robot/odom` | nav_msgs/Odometry | OUT | Robot position/velocity |
| `/lidar` | sensor_msgs/LaserScan | OUT | LiDAR distance scan |
| `/imu` | sensor_msgs/Imu | OUT | IMU acceleration/rotation |

## Performance Metrics

- **Inference Rate**: 30 Hz
- **Steering Range**: [-1.2, 1.2] radians
- **Max Linear Speed**: 0.5 m/s
- **Model Size**: 16.64 MB
- **Training Time**: ~2 minutes (10 epochs, 500 images)

## Troubleshooting

### Robot Not Moving
1. Check `/robot/cmd_vel` topic: `ros2 topic echo /robot/cmd_vel`
2. Verify differential drive controller is running
3. Check Gazebo physics: inspect `warehouse.world` physics tag

### Steering Predictions Stuck
1. Verify camera publishing: `ros2 topic echo /camera/image_raw`
2. Check CNN model loading: look for "[MODEL]" log messages
3. Run `debug_movement.py` to monitor steering angle values

### Gazebo Crashes
1. Reduce physics time step if needed (currently 0.001s)
2. Check world file syntax: `gz sdf <path>/warehouse.world`
3. Verify all plugins are properly compiled

## Future Improvements

- [ ] Add real camera dataset for transfer learning
- [ ] Implement multi-camera perception (front + sides)
- [ ] Add SLAM for global localization
- [ ] Implement obstacle avoidance with LiDAR
- [ ] Deploy on real robot hardware
- [ ] Add reinforcement learning fine-tuning

## Author & Citation

**Diksha** - MSc Robotics, University of Birmingham

For academic use, please cite:
```
@thesis{diksha2026cnnlane,
  title={CNN-Based Lane Following for Warehouse Robot Autonomy},
  author={Diksha},
  year={2026},
  school={University of Birmingham}
}
```

## License

MIT License - See LICENSE file for details

## Contact

GitHub: [@dikshajangam26](https://github.com/dikshajangam26)  
Email: dikshajangam26@gmail.com
