#!/usr/bin/env python3
"""
LiDAR Obstacle Detector Node for Autonomous Lane-Following Robot
- Subscribes to /scan (LiDAR data)
- Detects obstacles within safety radius
- Publishes obstacle positions and threat level
"""

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Float32, Bool
import math

class LidarObstacleDetector(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector')
        
        # Parameters
        self.declare_parameter('safety_radius', 0.5)  # meters
        self.declare_parameter('warning_radius', 1.0)  # meters
        self.declare_parameter('min_range', 0.1)  # ignore closer than this
        self.declare_parameter('max_range', 10.0)  # ignore farther than this
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        
        self.safety_radius = self.get_parameter('safety_radius').value
        self.warning_radius = self.get_parameter('warning_radius').value
        self.min_range = self.get_parameter('min_range').value
        self.max_range = self.get_parameter('max_range').value
        
        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.scan_callback,
            10
        )
        
        # Publishers
        self.obstacle_detected_pub = self.create_publisher(
            Bool,
            '/obstacle/detected',
            10
        )
        self.min_distance_pub = self.create_publisher(
            Float32,
            '/obstacle/min_distance',
            10
        )
        self.threat_level_pub = self.create_publisher(
            Float32,
            '/obstacle/threat_level',  # 0.0 (no threat) to 1.0 (critical)
            10
        )
        self.closest_obstacle_pub = self.create_publisher(
            PointStamped,
            '/obstacle/closest_point',
            10
        )
        
        # State
        self.last_scan = None
        self.obstacle_detected = False
        self.min_distance = float('inf')
        self.closest_angle = 0.0
        
        self.get_logger().info('LiDAR Obstacle Detector initialized')
        self.get_logger().info(f'Safety radius: {self.safety_radius}m, Warning radius: {self.warning_radius}m')

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan data"""
        self.last_scan = msg
        
        # Convert scan to cartesian coordinates
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges)
        
        # Filter invalid ranges
        valid_mask = (ranges >= self.min_range) & (ranges <= self.max_range) & np.isfinite(ranges)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            self.obstacle_detected = False
            self.min_distance = float('inf')
            self.publish_status()
            return
        
        # Find minimum distance
        self.min_distance = float(np.min(valid_ranges))
        min_idx = np.argmin(valid_ranges)
        self.closest_angle = valid_angles[min_idx]
        
        # Detect obstacles
        obstacles_in_safety = valid_ranges < self.safety_radius
        obstacles_in_warning = valid_ranges < self.warning_radius
        
        self.obstacle_detected = bool(np.any(obstacles_in_safety))
        
        # Calculate threat level (0 to 1)
        if self.min_distance < self.safety_radius:
            threat_level = 1.0  # Critical
        elif self.min_distance < self.warning_radius:
            threat_level = (self.warning_radius - self.min_distance) / (self.warning_radius - self.safety_radius)
        else:
            threat_level = 0.0  # No threat
        
        num_warning_obstacles = int(np.sum(obstacles_in_warning))
        
        self.get_logger().debug(
            f'Min distance: {self.min_distance:.2f}m, '
            f'Threat level: {threat_level:.2f}, '
            f'Obstacles in warning zone: {num_warning_obstacles}'
        )
        
        self.publish_status(threat_level)

    def publish_status(self, threat_level=0.0):
        """Publish obstacle detection status"""
        detected_msg = Bool()
        detected_msg.data = self.obstacle_detected
        self.obstacle_detected_pub.publish(detected_msg)
        
        dist_msg = Float32()
        dist_msg.data = float(self.min_distance) if self.min_distance != float('inf') else 0.0
        self.min_distance_pub.publish(dist_msg)
        
        threat_msg = Float32()
        threat_msg.data = threat_level
        self.threat_level_pub.publish(threat_msg)
        
        if self.last_scan and self.min_distance != float('inf'):
            closest_msg = PointStamped()
            closest_msg.header.stamp = self.last_scan.header.stamp
            closest_msg.header.frame_id = self.last_scan.header.frame_id
            closest_msg.point.x = self.min_distance * math.cos(self.closest_angle)
            closest_msg.point.y = self.min_distance * math.sin(self.closest_angle)
            closest_msg.point.z = 0.0
            self.closest_obstacle_pub.publish(closest_msg)


def main(args=None):
    rclpy.init(args=args)
    detector = LidarObstacleDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info('Node interrupted')
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
