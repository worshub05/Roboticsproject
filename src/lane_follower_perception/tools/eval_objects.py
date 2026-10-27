#!/usr/bin/env python3
"""
Check the LiDAR object detections (/objects) against Gazebo ground truth (/traffic/cars).

  python3 tools/eval_objects.py        (Ctrl+C prints the summary)

A car counts as VISIBLE when at least 3 LiDAR points of the current scan land on it
(a car hidden behind another car cannot be detected by any LiDAR method).
A visible car is DETECTED if an object centre is within 0.4 m of the true centre.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

HL, HW = 0.25, 0.125


class EvalObjects(Node):
    def __init__(self):
        super().__init__('eval_objects')
        self.truth, self.pose, self.scan = None, None, None
        self.errors, self.visible, self.detected, self.false_pos, self.frames = [], 0, 0, 0, 0
        self.create_subscription(Float32MultiArray, '/traffic/cars', lambda m: setattr(self, 'truth', list(m.data)), 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m), 10)
        self.create_subscription(Float32MultiArray, '/objects', self.objects_cb, 10)
        self.get_logger().info('Evaluating /objects against Gazebo ground truth... Ctrl+C for the summary')

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def objects_cb(self, msg):
        if self.truth is None or self.pose is None or self.scan is None:
            return
        x0, y0, yaw0 = self.pose
        r = np.asarray(self.scan.ranges, float)
        ang = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment + yaw0
        ok = np.isfinite(r)
        px, py = x0 + r[ok] * np.cos(ang[ok]), y0 + r[ok] * np.sin(ang[ok])
        objs = np.array(msg.data, float).reshape(-1, 6)
        used = set()
        for k in range(0, len(self.truth), 3):
            cx, cy, cyaw = self.truth[k:k + 3]
            c, s = math.cos(-cyaw), math.sin(-cyaw)
            lx, ly = (px - cx) * c - (py - cy) * s, (px - cx) * s + (py - cy) * c
            if int(np.sum((np.abs(lx) < HL + 0.05) & (np.abs(ly) < HW + 0.05))) < 3:
                continue                                    # not visible in this scan
            self.visible += 1
            if len(objs):
                e = np.hypot(objs[:, 0] - cx, objs[:, 1] - cy)
                j = int(np.argmin(e))
                if e[j] < 0.4:
                    self.detected += 1
                    self.errors.append(float(e[j]))
                    used.add(j)
        self.false_pos += len(objs) - len(used)
        self.frames += 1
        if self.frames % 150 == 0:
            self.report()

    def report(self):
        if not self.visible:
            self.get_logger().info('EVAL: no visible cars yet')
            return
        e = np.array(self.errors) * 100 if self.errors else np.array([0.0])
        self.get_logger().info(
            f'EVAL: {self.frames} scans, visible cars {self.visible}, detected {100 * self.detected / self.visible:.1f}%, '
            f'centre error median {np.median(e):.1f} cm / mean {e.mean():.1f} cm / 95% {np.percentile(e, 95):.1f} cm, '
            f'false detections {self.false_pos} ({self.false_pos / max(1, self.frames):.2f} per scan)')


def main():
    rclpy.init()
    node = EvalObjects()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.report()
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
