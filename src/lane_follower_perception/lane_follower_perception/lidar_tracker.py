#!/usr/bin/env python3
"""
LiDAR object tracker node.

Day 1 (this version): detection only. Every scan is clustered and a car-sized box is fitted to
each cluster (see lidar_objects.py). Detections are moved into the world (odom) frame using
the robot's pose and published as

  /objects   std_msgs/Float32MultiArray, 6 numbers per object: x, y, yaw, length, width, points

Day 2 adds a Kalman filter per object (speed and a stable ID), Day 3 adds prediction.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from .lidar_objects import detect_objects


class LidarTracker(Node):
    def __init__(self):
        super().__init__('lidar_tracker')
        self.declare_parameter('max_object_size', 1.0)
        self.max_size = self.get_parameter('max_object_size').value
        self.pose = None
        self.count = 0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.obj_pub = self.create_publisher(Float32MultiArray, '/objects', 10)
        self.get_logger().info('LiDAR tracker ready (day 1: clustering + box fitting)')

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def scan_callback(self, msg):
        if self.pose is None:
            return
        x0, y0, yaw0 = self.pose
        c, s = math.cos(yaw0), math.sin(yaw0)
        data = []
        objs = detect_objects(msg, self.max_size)
        for o in objs:
            wx, wy = x0 + c * o['x'] - s * o['y'], y0 + s * o['x'] + c * o['y']
            data += [wx, wy, (o['yaw'] + yaw0) % math.pi, o['length'], o['width'], float(o['n'])]
        self.obj_pub.publish(Float32MultiArray(data=[float(v) for v in data]))
        self.count += 1
        if self.count % 60 == 0:
            desc = ', '.join(f'({o["x"]:+.1f}, {o["y"]:+.1f}) {o["n"]} pts' for o in objs) or 'none'
            self.get_logger().info(f'TRACKER: {len(objs)} objects (robot frame): {desc}')


def main(args=None):
    rclpy.init(args=args)
    node = LidarTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
