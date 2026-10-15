from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from pathlib import Path

def generate_launch_description():
    ros2_ws = str(Path.home()) + '/ros2_ws'
    
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', ros2_ws + '/worlds/warehouse.world'],
        output='screen',
        name='gazebo'
    )
    
    cnn_node = Node(
        package='lane_follower_perception',
        executable='cnn_inference_node',
        name='cnn_inference_node',
        parameters=[{
            'model_path': ros2_ws + '/models/lane_cnn_retrained.pt',
            'camera_topic': '/camera/image_raw',
            'steering_topic': '/cmd_vel',
            'linear_speed': 0.5,
            'inference_rate': 30
        }],
        output='screen'
    )
    
    diff_drive = ExecuteProcess(
        cmd=['python3', ros2_ws + '/differential_drive_controller.py'],
        output='screen',
        name='diff_drive_controller'
    )
    
    return LaunchDescription([gazebo, cnn_node, diff_drive])
