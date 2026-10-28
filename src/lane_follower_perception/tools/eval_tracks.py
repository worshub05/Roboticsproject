#!/usr/bin/env python3
"""
Check the LiDAR detections (/objects) and tracks (/tracks) against Gazebo ground truth
(/traffic/cars_v: x, y, yaw, vx, vy per car).

  python3 tools/eval_tracks.py        (Ctrl+C prints the summary)

  visible     at least 3 LiDAR points of the current scan land on the car
  detected    a detection within 0.4 m of the true centre
  tracked     a confirmed track within 0.4 m of the true centre
  miss reason merged     (another car within 0.9 m, so the two blobs join)
              range edge (more than 9 m away)
              other
  ID switch   the track matched to a car changes within 1 s
  coasting    a car hidden from the LiDAR but within 9 m that still has a track within 0.5 m
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

HL, HW = 0.25, 0.125


class EvalTracks(Node):
    def __init__(self):
        super().__init__('eval_tracks')
        self.truth = self.pose = self.scan = self.objects = None
        self.n = dict(scans=0, visible=0, detected=0, tracked=0, switches=0, hidden=0, coasted=0)
        self.reasons = dict(merged=0, range_edge=0, other=0)
        self.err_det, self.err_trk, self.err_vel = [], [], []
        self.last_id = {}
        self.create_subscription(Float32MultiArray, '/traffic/cars_v', lambda m: setattr(self, 'truth', list(m.data)), 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m), 10)
        self.create_subscription(Float32MultiArray, '/objects', lambda m: setattr(self, 'objects', list(m.data)), 10)
        self.create_subscription(Float32MultiArray, '/tracks', self.tracks_cb, 10)
        self.get_logger().info('Evaluating detections and tracks against Gazebo ground truth... Ctrl+C for the summary')

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def tracks_cb(self, msg):
        if None in (self.truth, self.pose, self.scan, self.objects):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        x0, y0, yaw0 = self.pose
        r = np.asarray(self.scan.ranges, float)
        ang = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment + yaw0
        ok = np.isfinite(r)
        px, py = x0 + r[ok] * np.cos(ang[ok]), y0 + r[ok] * np.sin(ang[ok])
        objs = np.array(self.objects, float).reshape(-1, 6)
        trks = np.array(msg.data, float).reshape(-1, 8)
        cars = np.array(self.truth, float).reshape(-1, 5)
        for k, (cx, cy, cyaw, vx, vy) in enumerate(cars):
            c, s = math.cos(-cyaw), math.sin(-cyaw)
            lx, ly = (px - cx) * c - (py - cy) * s, (px - cx) * s + (py - cy) * c
            visible = int(np.sum((np.abs(lx) < HL + 0.05) & (np.abs(ly) < HW + 0.05))) >= 3
            dist = math.hypot(cx - x0, cy - y0)
            te, tj = 9.0, -1
            if len(trks):
                e = np.hypot(trks[:, 1] - cx, trks[:, 2] - cy)
                tj = int(np.argmin(e))
                te = float(e[tj])
            if visible:
                self.n['visible'] += 1
                de = float(np.min(np.hypot(objs[:, 0] - cx, objs[:, 1] - cy))) if len(objs) else 9.0
                if de < 0.4:
                    self.n['detected'] += 1
                    self.err_det.append(de)
                else:
                    others = np.hypot(np.delete(cars[:, 0], k) - cx, np.delete(cars[:, 1], k) - cy)
                    if len(others) and others.min() < 0.9:
                        self.reasons['merged'] += 1
                    elif dist > 9.0:
                        self.reasons['range_edge'] += 1
                    else:
                        self.reasons['other'] += 1
                if te < 0.4:
                    self.n['tracked'] += 1
                    self.err_trk.append(te)
                    self.err_vel.append(math.hypot(trks[tj, 3] - vx, trks[tj, 4] - vy))
                    tid = int(trks[tj, 0])
                    if k in self.last_id and self.last_id[k][0] != tid and now - self.last_id[k][1] < 1.0:
                        self.n['switches'] += 1
                    self.last_id[k] = (tid, now)
            elif dist < 9.0 and k in self.last_id and now - self.last_id[k][1] < 2.0:
                self.n['hidden'] += 1
                if te < 0.5:
                    self.n['coasted'] += 1
        self.n['scans'] += 1
        if self.n['scans'] % 300 == 0:
            self.report()

    def report(self):
        n = self.n
        if not n['visible']:
            self.get_logger().info('EVAL: no visible cars yet')
            return
        pct = lambda a, b: 100.0 * a / max(1, b)  # noqa: E731
        med = lambda v: float(np.median(v)) * 100 if v else 0.0  # noqa: E731
        p95 = lambda v: float(np.percentile(v, 95)) * 100 if v else 0.0  # noqa: E731
        self.get_logger().info(
            f'EVAL: {n["scans"]} scans, {n["visible"]} visible cars | detected {pct(n["detected"], n["visible"]):.1f}% '
            f'(misses: {self.reasons}) | tracked {pct(n["tracked"], n["visible"]):.1f}%')
        self.get_logger().info(
            f'EVAL: position error median/95%: detections {med(self.err_det):.1f}/{p95(self.err_det):.1f} cm, '
            f'tracks {med(self.err_trk):.1f}/{p95(self.err_trk):.1f} cm | speed error median/95% '
            f'{med(self.err_vel):.1f}/{p95(self.err_vel):.1f} cm/s | ID switches {n["switches"]} | '
            f'hidden cars kept by coasting {n["coasted"]}/{n["hidden"]}')


def main():
    rclpy.init()
    node = EvalTracks()
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
