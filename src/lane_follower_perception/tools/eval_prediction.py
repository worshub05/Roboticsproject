#!/usr/bin/env python3
"""
How good are the motion predictions? Every 0.5 s each confirmed track (/tracks) is predicted
1, 2 and 3 s ahead with two methods, and when that time comes the prediction is compared with
where the car really is (Gazebo ground truth, /traffic/cars_v):

  constant velocity   straight line from the current position and velocity
  lane following      along the lane at the current speed (lane_follower_perception/predict.py)

  python3 tools/eval_prediction.py        (Ctrl+C prints the summary)
"""
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32MultiArray

sys.path.insert(0, os.path.expanduser('~/ros2_ws/src/lane_follower_perception'))
from lane_follower_perception.predict import HORIZONS, predict_cv, predict_lane  # noqa: E402
from lane_follower_perception.track_map import on_bend, to_frenet, track_path  # noqa: E402


class EvalPrediction(Node):
    def __init__(self):
        super().__init__('eval_prediction', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.truth = None
        self.first_seen = {}
        self.pending = []
        self.errs = {}
        self.lane_ok = [0, 0]
        self.last = -1.0
        self.n = 0
        self.create_subscription(Float32MultiArray, '/traffic/cars_v', self.truth_cb, 10)
        self.create_subscription(Float32MultiArray, '/tracks', self.tracks_cb, 10)
        self.get_logger().info('Evaluating 1/2/3 s predictions against Gazebo ground truth... Ctrl+C for the summary')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def truth_cb(self, msg):
        self.truth = np.array(msg.data, float).reshape(-1, 5)
        t = self.now()
        for item in [p for p in self.pending if p[0] <= t]:
            self.pending.remove(item)
            _, k, h, cv, ln, bend = item
            cx, cy = self.truth[k, 0], self.truth[k, 1]
            for m, (px, py) in (('constant velocity', cv), ('lane following', ln)):
                self.errs.setdefault((m, h, bend), []).append(math.hypot(px - cx, py - cy))

    def tracks_cb(self, msg):
        if self.truth is None:
            return
        t = self.now()
        trks = np.array(msg.data, float).reshape(-1, 8)
        for tr in trks:
            self.first_seen.setdefault(int(tr[0]), t)
        if t - self.last < 0.5:
            return
        self.last = t
        for tid, x, y, vx, vy, *_ in trks:
            if t - self.first_seen[int(tid)] < 1.0:
                continue                                  # let the speed estimate settle first
            e = np.hypot(self.truth[:, 0] - x, self.truth[:, 1] - y)
            k = int(np.argmin(e))
            if e[k] > 0.4:
                continue
            cv = predict_cv(x, y, vx, vy)
            ln, lane, direction = predict_lane(x, y, vx, vy)
            tx, ty, _, tvx, tvy = self.truth[k]
            t_s, t_d = (float(a[0]) for a in to_frenet(tx, ty))
            true_lane = 0 if abs(t_d) < 0.25 else 1
            th_lane = track_path(t_s, 0.0 if true_lane == 0 else 0.5)[2]
            speed = tvx * math.cos(th_lane) + tvy * math.sin(th_lane)      # signed: + our way, - oncoming
            true_dir = 0 if abs(speed) < 0.03 else (1 if speed > 0 else -1)
            self.lane_ok[1] += 1
            self.lane_ok[0] += int(lane == true_lane and (true_dir == 0 or direction == true_dir))
            bend = any(on_bend(t_s + speed * hh * f) for hh in HORIZONS for f in (0.0, 0.5, 1.0))
            for h, a, b in zip(HORIZONS, cv, ln):
                self.pending.append((t + h, k, h, a, b, bend))
        self.n += 1
        if self.n % 60 == 0:
            self.report()

    def report(self):
        if not self.lane_ok[1]:
            self.get_logger().info('PRED: nothing evaluated yet')
            return
        self.get_logger().info(f'PRED: lane and direction correct for {100 * self.lane_ok[0] / self.lane_ok[1]:.1f}% '
                               f'of {self.lane_ok[1]} predictions')
        for bend in (False, True):
            for h in HORIZONS:
                parts = []
                for m in ('constant velocity', 'lane following'):
                    e = self.errs.get((m, h, bend))
                    if e:
                        e = np.array(e) * 100
                        parts.append(f'{m}: median {np.median(e):.1f} cm, 95% {np.percentile(e, 95):.1f} cm')
                if parts:
                    n = len(self.errs[('constant velocity', h, bend)])
                    self.get_logger().info(f'PRED: {"bend    " if bend else "straight"} {h:.0f} s ahead (n={n}): '
                                           + ' | '.join(parts))


def main():
    rclpy.init()
    node = EvalPrediction()
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
