#!/usr/bin/env python3
"""
LiDAR object tracker node.

Every scan:
  1. detection (lidar_objects.py): clustering + L-shape box fit with a car-size prior
  2. the boxes are moved into the world (odom) frame using the robot's pose
  3. tracking (mot.py): one constant-velocity Kalman filter per car, giving a smoothed
     position, a velocity and an ID that stays the same from scan to scan

Publishes (std_msgs/Float32MultiArray):
  /objects  6 numbers per detection:     x, y, yaw, length, width, points
  /tracks   8 numbers per confirmed car: id, x, y, vx, vy, yaw, length, width
  /predictions 9 numbers per confirmed car: id, lane (0 ours, 1 other, -1 off road),
               direction (+1 our way, -1 oncoming), then x, y after 1, 2 and 3 s (lane following)
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from .lidar_objects import detect_objects
from .mot import MultiObjectTracker
from .predict import predict_lane


class LidarTracker(Node):
    def __init__(self):
        super().__init__('lidar_tracker')
        self.declare_parameter('max_object_size', 1.0)
        self.declare_parameter('max_coast', 1.0)       # s a car is predicted while not seen
        self.max_size = self.get_parameter('max_object_size').value
        self.mot = MultiObjectTracker(max_coast=self.get_parameter('max_coast').value)
        self.pose = None
        self.count = 0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.obj_pub = self.create_publisher(Float32MultiArray, '/objects', 10)
        self.trk_pub = self.create_publisher(Float32MultiArray, '/tracks', 10)
        self.pred_pub = self.create_publisher(Float32MultiArray, '/predictions', 10)
        self.get_logger().info('LiDAR tracker ready (box fitting + Kalman filter tracking)')

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def scan_callback(self, msg):
        if self.pose is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x0, y0, yaw0 = self.pose
        c, s = math.cos(yaw0), math.sin(yaw0)
        dets, data = [], []
        for o in detect_objects(msg, self.max_size):
            wx, wy = x0 + c * o['x'] - s * o['y'], y0 + s * o['x'] + c * o['y']
            d = dict(o, x=wx, y=wy, yaw=(o['yaw'] + yaw0) % math.pi)
            dets.append(d)
            data += [wx, wy, d['yaw'], o['length'], o['width'], float(o['n'])]
        self.obj_pub.publish(Float32MultiArray(data=[float(v) for v in data]))

        tracks = self.mot.step(dets, t)
        out = []
        for tr in tracks:
            out += [float(tr.id), tr.x[0], tr.x[1], tr.x[2], tr.x[3], tr.yaw, tr.length, tr.width]
        self.trk_pub.publish(Float32MultiArray(data=[float(v) for v in out]))
        pred = []
        for tr in tracks:
            pts, lane, direction = predict_lane(*tr.x)
            pred += [float(tr.id), float(lane), float(direction)] + [float(v) for xy in pts for v in xy]
        self.pred_pub.publish(Float32MultiArray(data=pred))

        self.count += 1
        if self.count % 90 == 0:
            names = {(0, 1): 'ahead in our lane', (0, -1): 'in our lane (coming back?)', (1, -1): 'oncoming',
                     (1, 1): 'other lane, our way', (0, 0): 'our lane', (1, 0): 'other lane'}
            desc = []
            for tr in tracks:
                _, lane, direction = predict_lane(*tr.x)
                if tr.speed < 0.03:
                    direction = 0                      # stopped: no direction
                desc.append(f'#{tr.id} {math.hypot(tr.x[0] - x0, tr.x[1] - y0):.1f} m away at {tr.speed:.2f} m/s '
                            f'({names.get((lane, direction), "off road")})')
            desc = ', '.join(desc) or 'none'
            self.get_logger().info(f'TRACKER: {len(tracks)} cars tracked: {desc}')


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
