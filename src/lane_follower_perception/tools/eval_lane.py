#!/usr/bin/env python3
"""Measure lane keeping on the oval track: lateral offset from the lane centre, from odometry.

  python3 tools/eval_lane.py ours     # anticlockwise in our lane
  python3 tools/eval_lane.py other    # clockwise in the other lane
Prints running stats every 10 s and a summary on Ctrl+C.
"""
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from lane_follower_perception.collect_track_data import track_path, L, R


class LaneEval(Node):
    def __init__(self, lane):
        super().__init__('lane_eval')
        self.per = 2 * L + 2 * math.pi * R
        s = np.arange(0.0, self.per, 0.02)
        off = 0.0 if lane == 'ours' else 0.5
        pts = np.array([track_path(si, off) for si in s])
        self.px, self.py, self.pth = pts[:, 0], pts[:, 1], pts[:, 2]
        self.errs, self.dist, self.prev = [], 0.0, None
        self.last_print = time.time()
        self.create_subscription(Odometry, '/odom', self.callback, 50)
        self.get_logger().info(f'Evaluating lane "{lane}": stats every 10 s, Ctrl+C for the summary')

    def callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        i = int(np.argmin((self.px - x) ** 2 + (self.py - y) ** 2))
        th = self.pth[i]
        self.errs.append(-math.sin(th) * (x - self.px[i]) + math.cos(th) * (y - self.py[i]))
        if self.prev is not None:
            self.dist += math.hypot(x - self.prev[0], y - self.prev[1])
        self.prev = (x, y)
        if time.time() - self.last_print > 10:
            self.last_print = time.time()
            self.report()

    def report(self, final=False):
        if not self.errs:
            return
        a = np.abs(np.array(self.errs))
        print(f'{"SUMMARY" if final else "so far"}: {self.dist:.1f} m ({self.dist / self.per:.2f} laps) | '
              f'offset mean {a.mean():.3f} m, 95% {np.percentile(a, 95):.3f} m, max {a.max():.3f} m | '
              f'in lane {100 * (a < 0.25).mean():.1f}% | off road {100 * (a > 0.45).mean():.1f}%', flush=True)


def main():
    rclpy.init()
    node = LaneEval(sys.argv[1] if len(sys.argv) > 1 else 'ours')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.report(final=True)


if __name__ == '__main__':
    main()
