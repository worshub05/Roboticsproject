"""
Launch file for warehouse robot CNN lane following system
Starts: Gazebo simulation + CNN inference node + differential drive controller
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from pathlib import Path
import os


def generate_launch_description():
    """Generate launch description"""

    # Get package share directory
    ros2_ws = Path.home() / 'ros2_ws'
    worlds_dir = ros2_ws / 'worlds'
    models_dir = ros2_ws / 'models'
    scripts_dir = ros2_ws

    # Gazebo world file
    world_file = worlds_dir / 'warehouse.world'

    # Model paths
    model_path = models_dir / 'lane_cnn_retrained.pt'

    return LaunchDescription([
        # Gazebo simulation
        ExecuteProcess(
            cmd=['gazebo', '--verbose', str(world_file)],
            output='screen'
        ),

        # CNN inference node
        Node(
            package='lane_follower_perception',
            executable='cnn_inference_node',
            name='cnn_inference_node',
            output='screen',
            parameters=[
                {'model_path': str(model_path)},
                {'camera_topic': '/camera/image_raw'},
                {'steering_topic': '/cmd_vel'},
                {'linear_speed': 0.5},
                {'inference_rate': 30},
            ]
        ),

        # Differential drive controller
        ExecuteProcess(
            cmd=['python3', str(scripts_dir / 'differential_drive_controller.py')],
            output='screen'
        ),
    ])
