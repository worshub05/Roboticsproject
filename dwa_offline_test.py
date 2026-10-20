#!/usr/bin/env python3
"""
Offline closed-loop test of the DWA planner (no Gazebo).

Runs the real DWAObstacleAvoidance planning code against a simulated LiDAR
and a box the same size as the Gazebo obstacle, with a simple lane-keeping
controller standing in for the CNN. Prints the path and a summary.

Usage:  python3 dwa_offline_test.py [attr=value ...]
        e.g. python3 dwa_offline_test.py w_obstacle=4.0 safety=1.0
"""
import math
import sys
from types import SimpleNamespace

import numpy as np
import rclpy
from geometry_msgs.msg import Twist

sys.path.insert(0, '/home/shub_2605/ros2_ws/src/lane_follower_perception')
from lane_follower_perception.dwa_obstacle_avoidance import DWAObstacleAvoidance

BOX = (5.0, 6.5, -0.75, 0.75)       # xmin, xmax, ymin, ymax: same size as the Gazebo obstacle
N_RAYS = 720
ANGLES = -math.pi + np.arange(N_RAYS) * (2.0 * math.pi / N_RAYS)
DT = 1.0 / 30.0                       # planner runs at the CNN rate
ROBOT_RADIUS = 0.15


def raycast(x, y, th, max_range=10.0):
    """Range along each LiDAR ray from pose (x, y, th) to the box; inf when it misses."""
    dx = np.cos(th + ANGLES)
    dy = np.sin(th + ANGLES)
    with np.errstate(divide='ignore', invalid='ignore'):
        tx1, tx2 = (BOX[0] - x) / dx, (BOX[1] - x) / dx
        ty1, ty2 = (BOX[2] - y) / dy, (BOX[3] - y) / dy
    t_near = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    t_far = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (t_far >= t_near) & (t_far > 0)
    r = np.where(hit, np.maximum(t_near, 0.0), np.inf)
    r[r > max_range] = np.inf
    return r


def main():
    rclpy.init()
    node = DWAObstacleAvoidance()
    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        setattr(node, key, float(value))
        print(f'override: {key} = {value}')

    sent = {}
    node.cmd_pub.publish = lambda m: sent.update(v=m.linear.x, w=m.angular.z)
    node.active_pub.publish = lambda m: None

    x = y = th = 0.0
    t = 0.0
    step = 0
    min_clear = float('inf')
    max_dev = 0.0
    while t < 60.0 and x < BOX[1] + 1.0:
        r = raycast(x, y, th)
        node.scan = SimpleNamespace(ranges=r, angle_min=-math.pi,
                                    angle_increment=2.0 * math.pi / N_RAYS, range_min=0.1)
        desired = Twist()
        desired.linear.x = 0.5
        desired.angular.z = float(np.clip(-1.0 * y - 1.5 * th, -1.2, 1.2))  # lane-keeping stand-in
        node.desired = desired
        node.plan()

        v, w = sent['v'], sent['w']
        th += w * DT
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        t += DT
        step += 1
        nearest = float(np.min(r))
        min_clear = min(min_clear, nearest)
        max_dev = max(max_dev, abs(y))
        if step % 30 == 0:
            print(f't={t:5.1f}s  x={x:5.2f}  y={y:+.2f}  v={v:+.2f}  w={w:+.2f}  nearest={nearest:.2f}')

    passed = x >= BOX[1] + 1.0
    print('\n--- summary ---')
    print(f'passed obstacle : {passed}  (time {t:.1f} s)')
    print(f'min clearance   : {min_clear:.2f} m  (collision below {ROBOT_RADIUS} m)')
    print(f'max lane offset : {max_dev:.2f} m')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
