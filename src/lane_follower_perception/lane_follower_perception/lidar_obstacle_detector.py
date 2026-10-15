#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool

class LidarObstacleDetector(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_detector')
        
        self.declare_parameter('safety_radius', 0.5)
        self.declare_parameter('warning_radius', 1.0)
        
        self.safety_radius = self.get_parameter('safety_radius').value
        self.warning_radius = self.get_parameter('warning_radius').value
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.detected_pub = self.create_publisher(Bool, '/obstacle/detected', 10)
        self.min_dist_pub = self.create_publisher(Float32, '/obstacle/min_distance', 10)
        self.threat_pub = self.create_publisher(Float32, '/obstacle/threat_level', 10)
        
        self.get_logger().info('🚨 LiDAR Obstacle Detector initialized')

    def scan_callback(self, msg):
        ranges = msg.ranges
        min_distance = min(ranges) if ranges else float('inf')
        
        detected = min_distance < self.warning_radius
        threat_level = max(0.0, min(1.0, (self.warning_radius - min_distance) / 
                          (self.warning_radius - self.safety_radius)))
        
        self.detected_pub.publish(Bool(data=detected))
        self.min_dist_pub.publish(Float32(data=min_distance))
        self.threat_pub.publish(Float32(data=threat_level))

def main():
    rclpy.init()
    node = LidarObstacleDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
