#!/usr/bin/env python3
"""Curved-road test: oval track, no traffic. Spawn pose is configurable:
  ros2 launch lane_follower_perception track_phase2.launch.py                 # our lane, left bends
  ros2 launch lane_follower_perception track_phase2.launch.py x:=20 y:=0.5 yaw:=3.1416   # other lane, right bends
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('lane_follower_perception')
    world = os.path.join(pkg, 'worlds', 'track_oval.world')
    robot = os.path.join(pkg, 'robots', 'warehouse_robot_with_lidar.urdf')
    model = LaunchConfiguration('model')

    return LaunchDescription([
        DeclareLaunchArgument('model', default_value=os.path.expanduser('~/ros2_ws/models/lane_cnn_v3.pt')),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('drive', default_value='true',
                              description='false = spawn only, no CNN/road driver (for data collection)'),

        ExecuteProcess(cmd=['gzserver', '-s', 'libgazebo_ros_init.so',
                            '-s', 'libgazebo_ros_factory.so', world], output='screen'),
        ExecuteProcess(cmd=['gzclient'], output='screen'),

        TimerAction(period=6.0, actions=[
            Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-entity', 'warehouse_robot', '-file', robot,
                            '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                            '-z', '0.18', '-Y', LaunchConfiguration('yaw')]),
        ]),

        Node(package='lane_follower_perception', executable='cnn_inference_node.py', output='screen',
             parameters=[{'steering_topic': '/cmd_vel_desired', 'model_path': model}],
             condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(LaunchConfiguration('drive'))),
        Node(package='lane_follower_perception', executable='road_driver.py', output='screen',
             parameters=[{'use_sim_time': True}],
             condition=__import__('launch.conditions', fromlist=['IfCondition']).IfCondition(LaunchConfiguration('drive'))),
    ])
