#!/usr/bin/env python3
"""
Road demo: our robot drives as an autonomous car on a two-lane road with traffic.

  ros2 launch lane_follower_perception road_phase1.launch.py              # all traffic
  ros2 launch lane_follower_perception road_phase1.launch.py scenario:=follow

Scenarios: all | oncoming | follow | crossing
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('lane_follower_perception')
    world = os.path.join(pkg, 'worlds', 'road_with_traffic.world')
    robot = os.path.join(pkg, 'robots', 'warehouse_robot_with_lidar.urdf')
    model = os.path.expanduser('~/ros2_ws/models/lane_cnn_v2.pt')
    scenario = LaunchConfiguration('scenario')

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='all',
                              description='all | oncoming | follow | crossing'),

        ExecuteProcess(cmd=['gzserver', '-s', 'libgazebo_ros_init.so',
                            '-s', 'libgazebo_ros_factory.so', world], output='screen'),
        ExecuteProcess(cmd=['gzclient'], output='screen'),

        # spawn our car once Gazebo is up
        TimerAction(period=6.0, actions=[
            Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-entity', 'warehouse_robot', '-file', robot,
                            '-x', '0', '-y', '0', '-z', '0.18']),
        ]),

        # lateral control: CNN lane keeping
        Node(package='lane_follower_perception', executable='cnn_inference_node.py', output='screen',
             parameters=[{'steering_topic': '/cmd_vel_desired', 'model_path': model}]),

        # longitudinal control: adaptive cruise / stop from the LiDAR
        Node(package='lane_follower_perception', executable='road_driver.py', output='screen',
             parameters=[{'use_sim_time': True}]),

        # other cars
        Node(package='lane_follower_perception', executable='traffic_manager.py', output='screen',
             parameters=[{'use_sim_time': True, 'scenario': scenario}]),
    ])
