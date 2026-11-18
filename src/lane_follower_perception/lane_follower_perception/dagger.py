#!/usr/bin/env python3
"""
DAgger (Dataset Aggregation, Ross et al. 2011) for the multi-task lane CNN.

A network trained only on the expert's driving has a blind spot: once it drifts a little, it sees
images the expert never produced, and its errors grow. DAgger lets the NETWORK drive and asks the
expert what it should have done in every image the network itself caused:

    command = beta * expert + (1 - beta) * network      (beta = 1: expert drives, 0: network drives)
    label   = the expert's steering for that image (always)

If the robot gets more than `takeover` metres off the lane centre the expert takes over until it is
back within 0.1 m (an "intervention"), so data collection never leaves the road.

This node does both jobs:
  collect   count > 0: save every save_every-th image with the expert labels (same format as
            collect_track_data.py: img_<index>_<steer>.png + labels.csv), stop after count images
  evaluate  count = 0: drive for `duration` seconds and measure how well the NETWORK keeps the lane
            (beta 0); optional steering noise (noise_std) tests how well it recovers from a push

Every run appends one JSON line with its results to `report` and logs "DAGGER RESULT: ...".
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from .collect_track_data import L, R, track_path, wrap
from .multitask_cnn import load_multitask, predict


class DAgger(Node):
    def __init__(self):
        super().__init__('dagger')
        for k, v in dict(model_path=os.path.expanduser('~/ros2_ws/models/lane_mt_v1.pt'), lane='ours', beta=0.0,
                         count=0, duration=150.0, out_dir=os.path.expanduser('~/ros2_ws/dagger/data'),
                         report=os.path.expanduser('~/ros2_ws/dagger/report.jsonl'), tag='', speed=0.5,
                         save_every=2, takeover=0.25, noise_std=0.0, noise_time=1.5, gain_lateral=1.0,
                         gain_heading=1.5, warmup=3.0, seed=0).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.model_path, self.lane, self.beta = g('model_path'), g('lane'), float(g('beta'))
        self.count, self.duration = int(g('count')), float(g('duration'))
        self.out_dir, self.report, self.tag = g('out_dir'), g('report'), g('tag')
        self.v, self.save_every, self.takeover = float(g('speed')), int(g('save_every')), float(g('takeover'))
        self.noise_std, self.noise_time = float(g('noise_std')), float(g('noise_time'))
        self.k_lat, self.k_head, self.warmup = float(g('gain_lateral')), float(g('gain_heading')), float(g('warmup'))
        self.rng = np.random.default_rng(int(g('seed')))
        self.model = load_multitask(self.model_path)

        # reference path of the chosen lane (as in collect_track_data.py)
        per = 2 * L + 2 * math.pi * R
        s = np.arange(0.0, per, 0.02)
        off, flip = (0.0, False) if self.lane == 'ours' else (0.5, True)
        pts = np.array([track_path(si, off) for si in s])
        if flip:
            pts = pts[::-1].copy()
            pts[:, 2] = pts[:, 2] + math.pi
        self.px, self.py, self.pth = pts[:, 0], pts[:, 1], pts[:, 2]
        dth = np.array([wrap(b - a) for a, b in zip(self.pth, np.roll(self.pth, -1))])
        ds = np.hypot(np.diff(self.px, append=self.px[0]), np.diff(self.py, append=self.py[0]))
        self.kappa = dth / np.maximum(ds, 1e-6)
        k = np.concatenate([self.kappa, self.kappa])
        a, b = int(0.5 / 0.02), int(2.5 / 0.02)
        c = np.cumsum(np.concatenate([[0.0], k]))
        self.kappa_ahead = np.array([(c[i + b] - c[i + a]) / (b - a) for i in range(len(self.kappa))])

        if self.count > 0:
            os.makedirs(self.out_dir, exist_ok=True)
            self.csv_path = os.path.join(self.out_dir, 'labels.csv')
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w') as f:
                    f.write('file,steer,e_lat,e_head,kappa\n')
            self.saved = len([f for f in os.listdir(self.out_dir) if f.endswith('.png')])
        else:
            self.saved = 0
        self.start_saved = self.saved
        self.pose, self.speed = None, 0.0
        self.t0, self.t_prev, self.noise, self.frames = None, None, 0.0, 0
        self.intervening, self.interventions, self.t_intervening = False, 0, 0.0
        self.e2, self.n, self.e_max, self.disagree, self.distance = 0.0, 0, 0.0, 0.0, 0.0
        self.ms = []
        self.finished = False
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        what = f'collect {self.count} images' if self.count > 0 else f'evaluate for {self.duration:.0f} s'
        self.get_logger().info(f'DAGGER: {what}, model {os.path.basename(self.model_path)}, lane {self.lane}, '
                               f'beta {self.beta}, noise {self.noise_std} rad/s, takeover at {self.takeover} m')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.speed = msg.twist.twist.linear.x

    def expert(self):
        x, y, yaw = self.pose
        i = int(np.argmin((self.px - x) ** 2 + (self.py - y) ** 2))
        th = self.pth[i]
        e_lat = -math.sin(th) * (x - self.px[i]) + math.cos(th) * (y - self.py[i])
        e_head = wrap(yaw - th)
        w = self.v * self.kappa[i] - self.k_lat * e_lat - self.k_head * e_head
        return float(np.clip(w, -1.2, 1.2)), e_lat, e_head, float(self.kappa_ahead[i])

    def image_callback(self, msg):
        if self.pose is None or self.finished:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.t_prev is None else min(max(t - self.t_prev, 0.0), 0.3)
        self.t_prev = t
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        rgb = img if msg.encoding == 'rgb8' else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w_exp, e_lat, e_head, kappa = self.expert()
        t1 = time.perf_counter()
        w_net = float(np.clip(predict(self.model, rgb, passes=1)['steer'], -1.2, 1.2))
        self.ms.append((time.perf_counter() - t1) * 1000)

        # expert takes over far from the centre, hands back once close again
        if not self.intervening and abs(e_lat) > self.takeover:
            self.intervening = True
            self.interventions += 1
        elif self.intervening and abs(e_lat) < 0.1:
            self.intervening = False
        if dt > 0.0 and self.noise_std > 0.0:
            a = dt / self.noise_time
            self.noise += -a * self.noise + self.noise_std * math.sqrt(2 * a) * self.rng.normal()
        w = w_exp if self.intervening else self.beta * w_exp + (1 - self.beta) * w_net
        if not self.intervening:
            w += self.noise
        cmd = Twist()
        cmd.linear.x = self.v
        cmd.angular.z = float(np.clip(w, -1.2, 1.2))
        self.cmd_pub.publish(cmd)

        elapsed = t - self.t0
        if elapsed > self.warmup:                        # statistics after the start-up
            self.e2 += e_lat * e_lat
            self.n += 1
            self.e_max = max(self.e_max, abs(e_lat))
            self.disagree += abs(w_net - w_exp)
            self.distance += max(0.0, self.speed) * dt
            self.t_intervening += dt if self.intervening else 0.0

        self.frames += 1
        if self.count > 0 and self.frames % self.save_every == 0:
            name = f'img_{self.saved:06d}_{w_exp:.3f}.png'
            cv2.imwrite(os.path.join(self.out_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with open(self.csv_path, 'a') as f:
                f.write(f'{name},{w_exp:.4f},{e_lat:.4f},{e_head:.4f},{kappa:.4f}\n')
            self.saved += 1
            if (self.saved - self.start_saved) % 250 == 0:
                self.get_logger().info(f'DAGGER: saved {self.saved - self.start_saved}/{self.count} '
                                       f'(offset {e_lat:+.2f} m, interventions {self.interventions})')
        if self.frames % 100 == 0:
            self.get_logger().info('DAGGER STATS: ' + self.line(elapsed))
        done = (self.count > 0 and self.saved - self.start_saved >= self.count) or \
               (self.count == 0 and elapsed >= self.duration)
        if done:
            self.finish(elapsed)

    def result(self, elapsed):
        rms = math.sqrt(self.e2 / self.n) if self.n else float('nan')
        return dict(tag=self.tag, model=os.path.basename(self.model_path), lane=self.lane, beta=self.beta,
                    noise_std=self.noise_std, collected=self.saved - self.start_saved, time_s=round(elapsed, 1),
                    distance_m=round(self.distance, 1), rms_offset_cm=round(100 * rms, 2),
                    max_offset_cm=round(100 * self.e_max, 1), interventions=self.interventions,
                    interventions_per_100m=round(100 * self.interventions / max(self.distance, 1e-6), 2),
                    expert_share=round(self.t_intervening / max(elapsed - self.warmup, 1e-6), 3),
                    disagreement=round(self.disagree / max(self.n, 1), 4),
                    inference_ms=round(float(np.median(self.ms)) if self.ms else 0.0, 1))

    def line(self, elapsed):
        r = self.result(elapsed)
        return (f'{r["time_s"]:.0f} s, {r["distance_m"]:.1f} m, lane offset RMS {r["rms_offset_cm"]:.1f} cm '
                f'(max {r["max_offset_cm"]:.1f} cm), interventions {r["interventions"]} '
                f'({r["interventions_per_100m"]:.1f} per 100 m), network vs expert {r["disagreement"]:.3f} rad/s, '
                f'images saved {r["collected"]}')

    def finish(self, elapsed):
        self.finished = True
        self.cmd_pub.publish(Twist())
        r = self.result(elapsed)
        os.makedirs(os.path.dirname(self.report) or '.', exist_ok=True)
        with open(self.report, 'a') as f:
            f.write(json.dumps(r) + '\n')
        self.get_logger().info('DAGGER RESULT: ' + self.line(elapsed))


def main(args=None):
    rclpy.init(args=args)
    node = DAgger()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    try:
        if not node.finished and node.t0 is not None and node.t_prev is not None:
            node.finish(node.t_prev - node.t0)
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
