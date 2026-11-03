#!/usr/bin/env python3
"""
V2I controller on the car: listens to the traffic lights' broadcast (/v2i/spat), knows where it is
on the track from /odom, and publishes /v2i/speed_limit (Float32, -1 = no limit) so the road
driver stops at a red light (see v2i.py for the stop / go rules and the fail-safe).
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from .track_map import to_frenet
from .v2i import V2IPlanner, parse_spat

ROBOT_FRONT = 0.15


class V2IController(Node):
    def __init__(self):
        super().__init__('v2i_controller')
        for k, v in dict(b_comf=0.5, b_max=1.0, stop_margin=0.10, lookahead=15.0, timeout=1.0,
                         cruise=0.5, v_min=0.15, arrive_margin=1.0).items():
            self.declare_parameter(k, v)
        self.declare_parameter('glosa', True)
        g = lambda n: float(self.get_parameter(n).value)  # noqa: E731
        self.planner = V2IPlanner(g('b_comf'), g('b_max'), g('stop_margin'), g('lookahead'), g('timeout'),
                                  glosa=bool(self.get_parameter('glosa').value), cruise=g('cruise'),
                                  v_min=g('v_min'), arrive_margin=g('arrive_margin'))
        self.front, self.v = None, 0.0
        self.msgs, self.comm_ok = 0, None
        self.create_subscription(String, '/v2i/spat', self.spat_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(Float32, '/v2i/speed_limit', 10)
        self.create_timer(0.05, self.tick)
        p = self.planner
        self.get_logger().info(f'V2I controller ready: comfortable braking {p.b_comf} m/s2, at most {p.b_max} m/s2 '
                               f'for amber, stop {p.margin} m before the line, fail-safe after {p.timeout} s '
                               f'without messages; GLOSA {"on" if p.glosa else "off"}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spat_callback(self, msg):
        lights = parse_spat(msg.data)
        if lights:
            if not self.planner.map:
                self.get_logger().info('V2I: receiving ' + ', '.join(
                    f'light {l["id"]} (stop line at s={l["stop_s"]:.1f} m)' for l in lights))
            self.planner.receive(lights, self.now())
            self.msgs += 1

    def odom_callback(self, msg):
        s, _ = to_frenet(msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.front, self.v = float(s[0]) + ROBOT_FRONT, msg.twist.twist.linear.x

    def tick(self):
        t = self.now()
        ok = any(t - rx < self.planner.timeout for (_, _, rx) in self.planner.last.values())
        if self.planner.map and ok != self.comm_ok:
            self.get_logger().info('V2I: ✅ messages arriving' if ok else
                                   '⚠ V2I: no messages for over 1 s - fail-safe: treating lights as red')
            self.comm_ok = ok
        if self.front is None:
            return
        limit, _ = self.planner.update(self.front, self.v, t)
        for e in self.planner.events:
            self.get_logger().info('V2I: ' + e)
        self.planner.events.clear()
        self.pub.publish(Float32(data=float(limit)))


def main(args=None):
    rclpy.init(args=args)
    node = V2IController()
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
