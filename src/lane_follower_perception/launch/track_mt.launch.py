#!/usr/bin/env python3
"""
Oval track driven by the multi-task CNN (with uncertainty and camera health checks) + road driver.

  ros2 launch lane_follower_perception track_mt.launch.py
  ros2 launch lane_follower_perception track_mt.launch.py fault:=dark fault_after:=20
  ros2 launch lane_follower_perception track_mt.launch.py x:=20 y:=0.5 yaw:=3.1416    # other lane

fault: none | dark | noise | black | freeze
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('lane_follower_perception')
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='0'),
        DeclareLaunchArgument('y', default_value='0'),
        DeclareLaunchArgument('yaw', default_value='0'),
        DeclareLaunchArgument('model', default_value=os.path.expanduser('~/ros2_ws/models/lane_mt_v1.pt')),
        DeclareLaunchArgument('fault', default_value='none', description='none | dark | noise | black | freeze'),
        DeclareLaunchArgument('fault_after', default_value='20.0'),

        # world and robot only (no old CNN, no driver)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'track_phase2.launch.py')),
            launch_arguments={'drive': 'false', 'x': arg('x'), 'y': arg('y'), 'yaw': arg('yaw')}.items()),

        TimerAction(period=8.0, actions=[
            Node(package='lane_follower_perception', executable='multitask_inference_node.py', output='screen',
                 parameters=[{'use_sim_time': True, 'model_path': arg('model'),
                              'fault': arg('fault'), 'fault_after': ParameterValue(arg('fault_after'), value_type=float)}]),
            Node(package='lane_follower_perception', executable='road_driver.py', output='screen',
                 parameters=[{'use_sim_time': True}]),
        ]),
    ])
