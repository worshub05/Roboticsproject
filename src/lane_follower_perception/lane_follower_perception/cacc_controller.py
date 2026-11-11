#!/usr/bin/env python3
"""
CACC on our robot: follows the car in front at a short time gap using the LiDAR gap (/road/gap from
the road driver) plus that car's speed and acceleration received over V2V (/v2v/platoon).

Publishes /cacc/speed_cmd (Float32): the speed the road driver should drive (-1 = CACC not active).
Graceful degradation (see cacc.py): the older the newest V2V message (delay, lost messages), the
longer the time gap and the less the V2V data is used; with no usable V2V it drives on the LiDAR.
If the V2V messages stop completely (older than timeout) or the LiDAR sees no car in front, it
publishes -1 and the road driver falls back to its normal ACC with the long, safe 1.0 s time gap.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from .cacc import CACCLaw, parse_v2v, v2v_stamp


class CACCController(Node):
    def __init__(self):
        super().__init__('cacc_controller')
        for k, v in dict(r=0.4, h=0.3, k_a=0.8, k_p=0.6, k_d=2.0, front_id=1, timeout=2.0, max_gap=2.0).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.law = CACCLaw(r=float(g('r')), h=float(g('h')), k_a=float(g('k_a')), k_p=float(g('k_p')),
                           k_d=float(g('k_d')))
        self.front_id, self.timeout, self.max_gap = int(g('front_id')), float(g('timeout')), float(g('max_gap'))
        self.gap, self.gap_t, self.v = None, None, 0.0
        self.front, self.front_t, self.sent_t = None, None, None
        self.q_sum, self.q_n, self.h_max_seen = 0.0, 0, 0.0
        self.active, self.t_prev, self.count = None, None, 0
        self.create_subscription(Float32, '/road/gap', self.gap_callback, 10)
        self.create_subscription(String, '/v2v/platoon', self.v2v_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(Float32, '/cacc/speed_cmd', 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'CACC ready: gap = {self.law.r} m + {self.law.h} s x speed, gains k_a {self.law.ka}, '
                               f'k_p {self.law.kp}, k_d {self.law.kd}; following V2V car {self.front_id}, '
                               f'falls back to ACC after {self.timeout} s without V2V')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def gap_callback(self, msg):
        self.gap, self.gap_t = (float(msg.data) if msg.data >= 0 else None), self.now()

    def v2v_callback(self, msg):
        c = parse_v2v(msg.data).get(self.front_id)
        if c is not None:
            self.front, self.front_t = (float(c['v']), float(c['a'])), self.now()
            sent = v2v_stamp(msg.data)                       # when it was SENT (same clock as the sender)
            self.sent_t = sent if sent is not None and sent <= self.front_t else self.front_t

    def odom_callback(self, msg):
        self.v = msg.twist.twist.linear.x

    def tick(self):
        t = self.now()
        dt = 0.05 if self.t_prev is None else min(0.2, max(1e-3, t - self.t_prev))
        self.t_prev = t
        v2v_ok = self.front_t is not None and t - self.front_t < self.timeout
        gap_ok = self.gap is not None and self.gap_t is not None and t - self.gap_t < 0.5 and self.gap < self.max_gap
        ok = v2v_ok and gap_ok
        if ok != self.active:
            if ok:
                self.get_logger().info('CACC: ✅ active (V2V + LiDAR gap)')
            else:
                why = 'no V2V' if not v2v_ok else 'no car in front within %.1f m' % self.max_gap
                self.get_logger().info(f'CACC: ⚠ off ({why}) - road driver ACC (1.0 s time gap) takes over')
                self.law.v_ref = None
            self.active = ok
        cmd = -1.0
        if ok:
            cmd, _ = self.law.step(self.gap, self.v, self.front[0], self.front[1], dt, age=max(0.0, t - self.sent_t))
            self.q_sum, self.q_n = self.q_sum + self.law.q, self.q_n + 1
            self.h_max_seen = max(self.h_max_seen, self.law.h_eff)
        self.pub.publish(Float32(data=float(cmd)))
        self.count += 1
        if ok and self.count % 100 == 0:
            want = self.law.r + self.law.h_eff * self.v
            self.get_logger().info(f'CACC: gap {self.gap:.2f} m (want {want:.2f}), speed {self.v:.2f} m/s, '
                                   f'car in front {self.front[0]:.2f} m/s')
            self.get_logger().info(f'CACC STATS: V2V message age {self.law.age_f:.2f} s, degradation now '
                                   f'{100 * self.law.q:.0f}% (mean {100 * self.q_sum / self.q_n:.0f}%), time gap now '
                                   f'{self.law.h_eff:.2f} s (largest {self.h_max_seen:.2f} s)')


def main(args=None):
    rclpy.init(args=args)
    node = CACCController()
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
