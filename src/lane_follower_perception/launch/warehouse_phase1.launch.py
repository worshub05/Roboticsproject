#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('lane_follower_perception')
    world_file = os.path.join(pkg_dir, 'worlds', 'warehouse_with_obstacles.world')
    robot_urdf = os.path.join(pkg_dir, 'robots', 'warehouse_robot_with_lidar.urdf')

    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen'
    )

    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    cnn_node = Node(
        package='lane_follower_perception',
        executable='cnn_inference_node.py',
        output='screen',
        parameters=[{'steering_topic': '/cmd_vel_desired'}]
    )

    obstacle_node = Node(
        package='lane_follower_perception',
        executable='lidar_obstacle_detector.py',
        output='screen'
    )

    dwa_node = Node(
        package='lane_follower_perception',
        executable='dwa_obstacle_avoidance.py',
        output='screen'
    )

    ld = LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        cnn_node,
        obstacle_node,
        dwa_node,
    ])

    return ld
