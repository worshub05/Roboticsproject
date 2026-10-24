#!/usr/bin/env python3
"""
Collect CNN training images on the oval track, labelled by an expert that knows the track.

The robot is driven by the expert's steering plus a slowly wandering random nudge, so it
drifts off-centre the way a learned policy would. Every saved image is labelled with the
expert's CLEAN steering at that moment (how to get back to the lane centre), which gives
the network recovery examples (the idea behind DAgger and NVIDIA's shifted-camera data).

Parameters:
  lane        'ours'  = anticlockwise in our lane (left bends)
              'other' = clockwise in the other lane (right bends)
  count       images to save, then stop the robot
  out_dir     where images go: img_<index>_<steering>.png (same format as retrain_cnn.py)
"""
import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

L, R = 20.0, 5.0     # must match tools/make_track.py


def track_path(s, off):
    """Point and heading at arc length s along our lane centre, shifted 'off' metres to the left."""
    per = 2 * L + 2 * math.pi * R
    s %= per
    if s < L:
        x, y, th = s, 0.0, 0.0
    elif s < L + math.pi * R:
        a = (s - L) / R
        x, y, th = L + R * math.sin(a), R - R * math.cos(a), a
    elif s < 2 * L + math.pi * R:
        d = s - L - math.pi * R
        x, y, th = L - d, 2 * R, math.pi
    else:
        a = (s - 2 * L - math.pi * R) / R
        x, y, th = -R * math.sin(a), R + R * math.cos(a), math.pi + a
    return x - off * math.sin(th), y + off * math.cos(th), th


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class TrackDataCollector(Node):
    def __init__(self):
        super().__init__('collect_track_data')
        self.declare_parameter('lane', 'ours')
        self.declare_parameter('count', 3000)
        self.declare_parameter('out_dir', os.path.expanduser('~/ros2_ws/track_data'))
        self.declare_parameter('speed', 0.5)
        self.declare_parameter('save_every', 2)            # save every Nth camera frame
        self.declare_parameter('noise_std', 0.35)          # rad/s, size of the random nudge
        self.declare_parameter('noise_time', 1.5)          # s, how slowly the nudge wanders
        self.declare_parameter('gain_lateral', 1.0)
        self.declare_parameter('gain_heading', 1.5)
        self.declare_parameter('seed', 0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.lane = g('lane')
        self.count = int(g('count'))
        self.out_dir = g('out_dir')
        self.v = g('speed')
        self.save_every = int(g('save_every'))
        self.noise_std, self.noise_time = g('noise_std'), g('noise_time')
        self.k_lat, self.k_head = g('gain_lateral'), g('gain_heading')
        self.rng = np.random.default_rng(int(g('seed')))
        os.makedirs(self.out_dir, exist_ok=True)

        # Dense reference path for the chosen lane: points, headings, curvature (left turn > 0).
        per = 2 * L + 2 * math.pi * R
        s = np.arange(0.0, per, 0.02)
        off, flip = (0.0, False) if self.lane == 'ours' else (0.5, True)
        pts = np.array([track_path(si, off) for si in s])
        if flip:                                   # drive the other lane the other way round
            pts = pts[::-1].copy()
            pts[:, 2] = pts[:, 2] + math.pi
        self.px, self.py, self.pth = pts[:, 0], pts[:, 1], pts[:, 2]
        dth = np.array([wrap(b - a) for a, b in zip(self.pth, np.roll(self.pth, -1))])
        ds = np.hypot(np.diff(self.px, append=self.px[0]), np.diff(self.py, append=self.py[0]))
        self.kappa = dth / np.maximum(ds, 1e-6)

        self.pose = None
        self.noise = 0.0
        self.prev_t = None
        self.frames = 0
        self.saved = len([f for f in os.listdir(self.out_dir) if f.endswith('.png')])
        self.start_index = self.saved
        self.done = False

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info(f'Collecting {self.count} images on lane "{self.lane}" into {self.out_dir} '
                               f'({self.saved} already there)')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def expert(self):
        """Clean expert steering and the robot's lateral offset from the lane centre."""
        x, y, yaw = self.pose
        i = int(np.argmin((self.px - x) ** 2 + (self.py - y) ** 2))
        th = self.pth[i]
        e_lat = -math.sin(th) * (x - self.px[i]) + math.cos(th) * (y - self.py[i])   # + = left of centre
        e_head = wrap(yaw - th)
        w = self.v * self.kappa[i] - self.k_lat * e_lat - self.k_head * e_head
        return float(np.clip(w, -1.2, 1.2)), e_lat

    def image_callback(self, msg):
        if self.pose is None or self.done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else min(now - self.prev_t, 0.2)
        self.prev_t = now

        w_label, e_lat = self.expert()
        # Slowly wandering nudge (Ornstein-Uhlenbeck); pulled back hard if the robot nears the road edge.
        if dt > 0.0:
            a = dt / self.noise_time
            self.noise += -a * self.noise + self.noise_std * math.sqrt(2 * a) * self.rng.normal()
        if abs(e_lat) > 0.3:
            self.noise = 0.0

        cmd = Twist()
        cmd.linear.x = self.v
        cmd.angular.z = float(np.clip(w_label + self.noise, -1.2, 1.2))
        self.cmd_pub.publish(cmd)

        self.frames += 1
        if self.frames % self.save_every:
            return
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)     # retrain_cnn.py reads with cv2 (BGR)
        cv2.imwrite(os.path.join(self.out_dir, f'img_{self.saved:06d}_{w_label:.3f}.png'), img)
        self.saved += 1

        n = self.saved - self.start_index
        if n % 250 == 0:
            self.get_logger().info(f'saved {n}/{self.count}  (offset {e_lat:+.2f} m, label {w_label:+.2f})')
        if n >= self.count:
            self.done = True
            self.cmd_pub.publish(Twist())
            self.get_logger().info(f'✅ done: {n} images in {self.out_dir}. Press Ctrl+C.')


def main(args=None):
    rclpy.init(args=args)
    node = TrackDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass  # ROS already shut down by Ctrl+C


if __name__ == '__main__':
    main()
