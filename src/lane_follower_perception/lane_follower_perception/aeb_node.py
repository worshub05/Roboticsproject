#!/usr/bin/env python3
"""
AEB node: time-to-collision emergency braking on the raw LiDAR scan (see aeb.py).

Subscribes /scan and /odom. Publishes
  /safety/aeb_limit  Float32  speed limit (-1 = none, 0 = full emergency brake)
  /safety/aeb        String   OFF | WARNING | PARTIAL | FULL
and logs every activation plus running statistics ("AEB STATS: ...").
Parameter enforce:=false only monitors (publishes -1), for a with / without comparison.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

from .aeb import AEB, path_distance
from .cacc import parse_v2v
from .track_map import PER, to_frenet


class AEBNode(Node):
    def __init__(self):
        super().__init__('aeb')
        for k, v in dict(enforce=True, warn_ttc=1.5, brake_ttc=1.0, full_ttc=0.6, min_distance=0.08,
                         half_width=0.2, front_offset=0.15, cooperative=True, v2v_id=1).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.enforce = bool(g('enforce'))
        self.half_width, self.front = float(g('half_width')), float(g('front_offset'))
        self.aeb = AEB(warn_ttc=float(g('warn_ttc')), brake_ttc=float(g('brake_ttc')),
                       full_ttc=float(g('full_ttc')), min_distance=float(g('min_distance')))
        self.v, self.w = 0.0, 0.0
        # cooperative AEB: closing speed from the V2V speed of the car in front (if it is the object seen)
        self.cooperative, self.v2v_id = bool(g('cooperative')), int(g('v2v_id'))
        self.front_s, self.v2v, self.coop_scans, self.coop_on = None, None, 0, False
        self.create_subscription(String, '/v2v/platoon', self.v2v_callback, 10)
        self.t_prev, self.event_start, self.event_peak = None, None, 'OFF'
        self.event_min_ttc, self.event_min_d = math.inf, math.inf
        self.stats = dict(t=0.0, n_warn=0, n_partial=0, n_full=0, t_braking=0.0, min_ttc=math.inf)
        self.count = 0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.limit_pub = self.create_publisher(Float32, '/safety/aeb_limit', 10)
        self.state_pub = self.create_publisher(String, '/safety/aeb', 10)
        a = self.aeb
        self.get_logger().info(f'AEB ready ({"enforcing" if self.enforce else "monitoring only"}): '
                               f'warning TTC < {a.warn_ttc} s, partial brake < {a.brake_ttc} s, '
                               f'full brake < {a.full_ttc} s or closer than {a.min_distance} m')

    def odom_callback(self, msg):
        self.v, self.w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        s, _ = to_frenet(msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.front_s = (float(s[0]) + self.front) % PER

    def v2v_callback(self, msg):
        c = parse_v2v(msg.data).get(self.v2v_id)
        if c is not None:
            self.v2v = (float(c['s']), float(c['v']), self.get_clock().now().nanoseconds * 1e-9)

    def v2v_closing(self, d):
        """Closing speed from V2V if the LiDAR object is the V2V car in front (positions agree), else None."""
        if not self.cooperative or self.v2v is None or self.front_s is None or not math.isfinite(d):
            return None
        s_car, v_car, t_rx = self.v2v
        if self.get_clock().now().nanoseconds * 1e-9 - t_rx > 0.5:
            return None
        expected = (s_car - 0.25 - self.front_s) % PER                # V2V position -> gap to its back
        return self.v - v_car if abs(expected - d) < 0.25 else None

    def scan_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.t_prev is None else min(0.5, max(0.0, t - self.t_prev))
        self.t_prev = t
        r = np.asarray(msg.ranges, float)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max - 0.02)
        x, y = r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])
        curvature = self.w / self.v if abs(self.v) > 0.05 else 0.0
        curvature = float(np.clip(curvature, -2.0, 2.0))
        d = path_distance(x, y, curvature, self.half_width, self.front)
        prev = self.aeb.state
        closing = self.v2v_closing(d)
        if (closing is not None) != self.coop_on:
            self.coop_on = closing is not None
            self.get_logger().info('AEB: closing speed from V2V (cooperative)' if self.coop_on else
                                   'AEB: closing speed from the LiDAR')
        self.coop_scans += closing is not None
        state, limit, ttc = self.aeb.step(d, self.v, t, closing)

        st = self.stats
        st['t'] += dt
        if state in ('PARTIAL', 'FULL'):
            st['t_braking'] += dt
        if state != 'OFF':
            st['min_ttc'] = min(st['min_ttc'], ttc)
            self.event_min_ttc, self.event_min_d = min(self.event_min_ttc, ttc), min(self.event_min_d, d)
            if AEB.STATES.index(state) > AEB.STATES.index(self.event_peak):   # event reached a new level
                st['n_' + {'WARNING': 'warn', 'PARTIAL': 'partial', 'FULL': 'full'}[state]] += 1
                self.event_peak = state
        if prev == 'OFF' and state != 'OFF':
            self.event_start = t
            self.get_logger().info(f'AEB: ⚠ {state}: object {d:.2f} m ahead in our path, '
                                   f'TTC {ttc:.2f} s, our speed {self.v:.2f} m/s')
        elif prev != state and state in ('PARTIAL', 'FULL'):
            self.get_logger().info(f'AEB: ⚠ {state} brake: object {d:.2f} m ahead, TTC {ttc:.2f} s, '
                                   f'our speed {self.v:.2f} m/s')
        elif prev != 'OFF' and state == 'OFF':
            self.get_logger().info(f'AEB: ✅ released after {t - self.event_start:.1f} s (worst: {self.event_peak}, '
                                   f'TTC {self.event_min_ttc:.2f} s, closest {self.event_min_d:.2f} m)')
            self.event_peak, self.event_min_ttc, self.event_min_d = 'OFF', math.inf, math.inf

        self.limit_pub.publish(Float32(data=float(limit if self.enforce else -1.0)))
        self.state_pub.publish(String(data=state))
        self.count += 1
        if self.count % 50 == 0:
            mt = f'{st["min_ttc"]:.2f} s' if math.isfinite(st['min_ttc']) else 'n/a'
            self.get_logger().info(
                f'AEB STATS: checked {st["t"]:.0f} s, warnings {st["n_warn"]}, partial brakes {st["n_partial"]}, '
                f'full brakes {st["n_full"]}, braking {st["t_braking"]:.1f} s, smallest TTC {mt}, '
                f'closing speed from V2V in {100 * self.coop_scans / self.count:.0f}% of scans')


def main(args=None):
    rclpy.init(args=args)
    node = AEBNode()
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
