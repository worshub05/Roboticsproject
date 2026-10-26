#!/usr/bin/env python3
"""
Multi-task CNN lane keeping with self-monitoring.

Every cycle the camera image goes through two independent checks before the car trusts it:

  1. Camera health (no neural network): is the image too dark, too bright, washed out (no
     contrast), frozen while the car is moving, or not arriving at all?
  2. Network confidence (Monte Carlo dropout): the CNN runs N times with dropout on; a large
     spread in its steering answers means it is unsure about this image.

Status and what the car does:
  OK            normal driving at linear_speed
  UNCERTAIN     the CNN is unsure  -> slow down to slow_factor x linear_speed
  CAMERA_FAULT  the image can't be trusted -> ask for speed 0 (the road driver brakes)
                and stay stopped until the camera has looked healthy for recover_time seconds

Publishes:
  steering_topic (Twist)   speed + steering for the road driver (default /cmd_vel_desired)
  /lane/offset, /lane/heading, /lane/curvature, /lane/uncertainty (Float32)
  /lane/brightness, /lane/contrast (Float32), /lane/status (String)

Fault injection for experiments (parameters):
  fault        none | dark | noise | black | freeze
  fault_after  seconds after the first image before the fault starts
"""
import math
import os
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from .multitask_cnn import load_multitask, predict


class MultiTaskInferenceNode(Node):
    def __init__(self):
        super().__init__('multitask_inference_node')
        self.declare_parameter('model_path', os.path.expanduser('~/ros2_ws/models/lane_mt_v1.pt'))
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('steering_topic', '/cmd_vel_desired')
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('rate', 10.0)                 # Hz
        self.declare_parameter('passes', 10)                 # Monte Carlo dropout passes
        self.declare_parameter('uncertainty_alarm', 0.071)   # rad/s, from train_multitask_cnn.py
        self.declare_parameter('uncertainty_clear', 0.060)   # rad/s, hysteresis
        self.declare_parameter('uncertainty_filter', 0.1)    # 0..1, smoothing of the spread
        self.declare_parameter('slow_factor', 0.5)
        self.declare_parameter('min_brightness', 0.08)       # mean pixel value, 0..1
        self.declare_parameter('max_brightness', 0.95)
        self.declare_parameter('min_contrast', 0.03)         # pixel standard deviation, 0..1
        self.declare_parameter('max_noise', 0.05)            # estimated sensor noise (std), 0..1
        self.declare_parameter('freeze_time', 2.0)           # s of identical frames while moving
        self.declare_parameter('stale_time', 0.5)            # s without a new image
        self.declare_parameter('recover_time', 2.0)          # s healthy before driving again
        self.declare_parameter('fault', 'none')
        self.declare_parameter('fault_after', 20.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.v_nom = g('linear_speed')
        self.passes = int(g('passes'))
        self.u_alarm, self.u_clear, self.u_alpha = g('uncertainty_alarm'), g('uncertainty_clear'), g('uncertainty_filter')
        self.slow = g('slow_factor')
        self.b_min, self.b_max, self.c_min = g('min_brightness'), g('max_brightness'), g('min_contrast')
        self.n_max = g('max_noise')
        self.noise_level = 0.0
        self.freeze_time, self.stale_time, self.recover_time = g('freeze_time'), g('stale_time'), g('recover_time')
        self.fault, self.fault_after = g('fault'), g('fault_after')

        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.model = load_multitask(g('model_path'))
        self.get_logger().info(f'Loaded multi-task CNN {g("model_path")} ({self.passes} dropout passes)')

        self.image = None          # latest camera image (RGB uint8)
        self.image_t = None        # time it arrived
        self.first_t = None
        self.frozen_img = None
        self.prev_small = None
        self.same_since = None
        self.frozen_ref = None     # the stuck frame, once a freeze has been detected
        self.v_ego = 0.0
        self.u_filt = None
        self.status = 'OK'
        self.healthy_since = None
        self.count = 0
        self.ms = []

        self.create_subscription(Image, g('camera_topic'), self.image_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, g('steering_topic'), 10)
        self.pubs = {n: self.create_publisher(Float32, '/lane/' + n, 10)
                     for n in ('offset', 'heading', 'curvature', 'uncertainty', 'brightness', 'contrast', 'noise')}
        self.status_pub = self.create_publisher(String, '/lane/status', 10)
        self.create_timer(1.0 / g('rate'), self.timer_callback)
        if self.fault != 'none':
            self.get_logger().info(f'FAULT TEST: "{self.fault}" will start {self.fault_after:.0f} s after the first image')

    # ------------------------------------------------------------------ inputs
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg):
        self.v_ego = msg.twist.twist.linear.x

    def image_callback(self, msg):
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        self.image, self.image_t = np.ascontiguousarray(img), self.now()

    def timer_callback(self):
        if self.image is not None:
            self.step(self.image, self.image_t, self.now())

    # ----------------------------------------------------------- fault tests
    def inject(self, img, now):
        if self.first_t is None:
            self.first_t = now
        if self.fault == 'none' or now - self.first_t < self.fault_after:
            return img
        if self.fault == 'dark':
            return (img * 0.15).astype(np.uint8)
        if self.fault == 'black':
            return np.zeros_like(img)
        if self.fault == 'noise':
            noisy = img.astype(np.float32) + np.random.normal(0, 70, img.shape)
            return np.clip(noisy, 0, 255).astype(np.uint8)
        if self.fault == 'freeze':
            if self.frozen_img is None:
                self.frozen_img = img.copy()
            return self.frozen_img
        return img

    # ---------------------------------------------------------- camera health
    def camera_check(self, img, img_t, now):
        """Returns (reason or None, brightness, contrast)."""
        small = img[::4, ::4].astype(np.float32) / 255.0
        bright, contrast = float(small.mean()), float(small.std())
        # Pixel noise estimate (Immerkaer 1996): a 3x3 filter that cancels smooth image content
        # and edges' first-order change, so what is left is mostly sensor noise.
        grey = img[::2, ::2].astype(np.float32).mean(axis=2) / 255.0
        lap = (grey[:-2, :-2] - 2 * grey[:-2, 1:-1] + grey[:-2, 2:]
               - 2 * grey[1:-1, :-2] + 4 * grey[1:-1, 1:-1] - 2 * grey[1:-1, 2:]
               + grey[2:, :-2] - 2 * grey[2:, 1:-1] + grey[2:, 2:])
        self.noise_level = float(math.sqrt(math.pi / 2) * np.abs(lap).mean() / 6)
        if now - img_t > self.stale_time:
            return 'no new images', bright, contrast
        if self.prev_small is not None and float(np.abs(small - self.prev_small).mean()) < 1e-3 \
                and abs(self.v_ego) > 0.1:
            self.same_since = now if self.same_since is None else self.same_since
        else:
            self.same_since = None
        self.prev_small = small
        if bright < self.b_min:
            return f'too dark ({bright:.2f})', bright, contrast
        if bright > self.b_max:
            return f'too bright ({bright:.2f})', bright, contrast
        if contrast < self.c_min:
            return f'no contrast ({contrast:.3f})', bright, contrast
        if self.noise_level > self.n_max:
            return f'too noisy ({self.noise_level:.3f})', bright, contrast
        if self.same_since is not None and now - self.same_since > self.freeze_time:
            self.frozen_ref = small          # remember the stuck frame: stays a fault even after we stop
        if self.frozen_ref is not None:
            if float(np.abs(small - self.frozen_ref).mean()) < 1e-3:
                return 'image frozen', bright, contrast
            self.frozen_ref = None
        return None, bright, contrast

    # -------------------------------------------------------------- main step
    def step(self, img, img_t, now):
        img = self.inject(img, now)
        reason, bright, contrast = self.camera_check(img, img_t, now)

        t0 = time.perf_counter()
        p = predict(self.model, img, passes=self.passes)
        self.ms.append((time.perf_counter() - t0) * 1000)
        u = p['steer_std']
        self.u_filt = u if self.u_filt is None else (1 - self.u_alpha) * self.u_filt + self.u_alpha * u

        # status with hysteresis; a camera fault latches until healthy for recover_time
        if reason is not None:
            status, self.healthy_since = 'CAMERA_FAULT', None
        elif self.status == 'CAMERA_FAULT':
            self.healthy_since = now if self.healthy_since is None else self.healthy_since
            status = 'CAMERA_FAULT' if now - self.healthy_since < self.recover_time else 'OK'
        else:
            status = self.status
        if status != 'CAMERA_FAULT':
            if self.u_filt > self.u_alarm:
                status = 'UNCERTAIN'
            elif self.u_filt < self.u_clear:
                status = 'OK'
            elif status not in ('OK', 'UNCERTAIN'):
                status = 'OK'

        if status != self.status:
            why = reason if reason else f'uncertainty {self.u_filt:.3f} rad/s'
            self.get_logger().info(f'LANE: {self.status} -> {status} ({why})')
            self.status = status

        speed = {'OK': self.v_nom, 'UNCERTAIN': self.slow * self.v_nom, 'CAMERA_FAULT': 0.0}[status]
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(np.clip(p['steer'], -1.2, 1.2)) if status != 'CAMERA_FAULT' else 0.0
        self.cmd_pub.publish(cmd)

        for name, val in (('offset', p['e_lat']), ('heading', p['e_head']), ('curvature', p['kappa']),
                          ('uncertainty', self.u_filt), ('brightness', bright), ('contrast', contrast),
                          ('noise', self.noise_level)):
            self.pubs[name].publish(Float32(data=float(val)))
        self.status_pub.publish(String(data=status))

        self.count += 1
        if self.count % 30 == 0:
            self.get_logger().info(
                f'LANE: {status} offset {p["e_lat"]*100:+.1f} cm, heading {math.degrees(p["e_head"]):+.1f} deg, '
                f'curvature {p["kappa"]:+.3f} 1/m, uncertainty {self.u_filt:.3f}, '
                f'brightness {bright:.2f}, contrast {contrast:.3f}, noise {self.noise_level:.3f}, {np.mean(self.ms[-30:]):.0f} ms/frame')
        return status, cmd


def main(args=None):
    rclpy.init(args=args)
    node = MultiTaskInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
