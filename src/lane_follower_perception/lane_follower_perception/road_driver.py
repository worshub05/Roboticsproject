#!/usr/bin/env python3
"""
Road driver: lane keeping from the CNN plus LiDAR-based speed control.

Lateral:      steering comes from the CNN (/cmd_vel_desired).
Longitudinal: the LiDAR is checked inside a corridor that follows our own lane ahead.
              With the multi-task CNN running, the corridor bends with the predicted road
              curvature (/lane/curvature) and is shifted by the predicted lane offset and
              heading error (/lane/offset, /lane/heading), so a car ahead on a bend stays
              inside it. Without those topics it falls back to a straight corridor.
              - nothing in the corridor  -> cruise at the CNN's set speed
              - a car ahead              -> adaptive cruise control: follow it at a
                                            constant time gap (gap = d0 + h * v)
              - something very close     -> stop (e.g. a car crossing in front)
              Objects outside the corridor, such as oncoming cars in the other
              lane, are ignored.

Publishes /cmd_vel plus /road/gap, /road/lead_speed and /road/state for logging.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String


class RoadDriver(Node):
    def __init__(self):
        super().__init__('road_driver')

        self.declare_parameter('max_speed', 0.5)            # m/s, cruise speed cap
        self.declare_parameter('corridor_half_width', 0.3)  # m, our lane: lines at +/-0.25
        self.declare_parameter('lookahead', 4.0)            # m, how far ahead we watch
        self.declare_parameter('front_offset', 0.15)        # m, LiDAR to front bumper
        self.declare_parameter('standstill_gap', 0.5)       # m, d0: gap kept when stopped
        self.declare_parameter('time_gap', 1.0)             # s, h: extra gap per m/s of speed
        self.declare_parameter('gap_gain', 0.6)             # 1/s, how hard we close a gap error
        self.declare_parameter('stop_distance', 0.3)        # m, hard stop if anything is closer
        self.declare_parameter('max_accel', 0.4)            # m/s^2
        self.declare_parameter('max_decel', 1.5)            # m/s^2
        self.declare_parameter('speed_filter', 0.15)         # 0..1, smoothing of the lead-speed estimate
        self.declare_parameter('curved_corridor', True)     # False = always straight (for comparison)
        self.declare_parameter('lane_timeout', 0.5)         # s, lane topics older than this are ignored
        self.declare_parameter('lane_filter', 0.3)          # 0..1, smoothing of curvature / offset / heading

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.max_speed = g('max_speed')
        self.half_width = g('corridor_half_width')
        self.lookahead = g('lookahead')
        self.front_offset = g('front_offset')
        self.d0 = g('standstill_gap')
        self.h = g('time_gap')
        self.k_gap = g('gap_gain')
        self.stop_distance = g('stop_distance')
        self.max_accel = g('max_accel')
        self.max_decel = g('max_decel')
        self.alpha = g('speed_filter')
        self.curved = bool(g('curved_corridor'))
        self.lane_timeout = g('lane_timeout')
        self.lane_alpha = g('lane_filter')
        self.lane = {'curvature': 0.0, 'offset': 0.0, 'heading': 0.0}
        self.lane_t = None

        self.desired = Twist()
        self.scan = None
        self.v_ego = None          # measured speed from odometry
        self.v_cmd = 0.0           # last commanded speed
        self.prev_t = None
        self.prev_gap = None
        self.lead_speed = None
        self.state = 'CRUISE'
        self.count = 0

        self.create_subscription(Twist, '/cmd_vel_desired', self.desired_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        for name in self.lane:
            self.create_subscription(Float32, '/lane/' + name, lambda m, n=name: self.lane_callback(n, m), 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gap_pub = self.create_publisher(Float32, '/road/gap', 10)
        self.lead_pub = self.create_publisher(Float32, '/road/lead_speed', 10)
        self.state_pub = self.create_publisher(String, '/road/state', 10)
        # RSS safety layer (rss_monitor): never drive faster than its speed limit
        self.rss_limit, self.rss_t = None, None
        self.create_subscription(Float32, '/safety/speed_limit', self.rss_callback, 10)

        self.get_logger().info('Road driver ready: CNN steering + LiDAR adaptive cruise control '
                               f'({"curved" if self.curved else "straight"} corridor)')

    # ------------------------------------------------------------------ inputs
    def scan_callback(self, msg):
        self.scan = msg

    def odom_callback(self, msg):
        self.v_ego = msg.twist.twist.linear.x

    def rss_callback(self, msg):
        self.rss_limit = None if msg.data < 0 else float(msg.data)
        self.rss_t = self.get_clock().now().nanoseconds * 1e-9

    def lane_callback(self, name, msg):
        a = self.lane_alpha
        self.lane[name] = (1 - a) * self.lane[name] + a * float(msg.data)
        if name == 'curvature':
            self.lane_t = self.get_clock().now().nanoseconds * 1e-9

    def lane_shape(self, now):
        """(curvature, offset, heading) of our lane, or zeros if not known / switched off."""
        if not self.curved or self.lane_t is None or now - self.lane_t > self.lane_timeout:
            return 0.0, 0.0, 0.0
        return self.lane['curvature'], self.lane['offset'], self.lane['heading']

    def desired_callback(self, msg):
        self.desired = msg
        now = self.get_clock().now().nanoseconds * 1e-9
        self.control(now)

    # ------------------------------------------------------------- perception
    def gap_ahead(self, kappa=0.0, e_lat=0.0, e_head=0.0):
        """
        Distance along our lane from the front bumper to the nearest LiDAR return inside the lane
        corridor. The corridor is the lane centre ahead (a circular arc of curvature kappa, or a
        straight line) +/- corridor_half_width; e_lat / e_head place the robot relative to it.
        """
        if self.scan is None:
            return math.inf
        r = np.asarray(self.scan.ranges, dtype=float)
        angles = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment
        ok = np.isfinite(r) & (r > self.scan.range_min)
        x = r[ok] * np.cos(angles[ok])
        y = r[ok] * np.sin(angles[ok])
        # points in the lane frame: origin on the lane centre beside us, X along the lane, Y to the left
        c, s_ = math.cos(e_head), math.sin(e_head)
        X = x * c - y * s_
        Y = e_lat + x * s_ + y * c
        if abs(kappa) < 0.02:                      # (almost) straight
            along, lateral = X, Y
        else:                                      # arc: centre of the bend at (0, 1/kappa)
            Rc = 1.0 / kappa
            ang = np.arctan2(X, abs(Rc) - np.sign(Rc) * Y)   # angle swept around the bend
            along = abs(Rc) * ang
            lateral = np.sign(Rc) * (abs(Rc) - np.hypot(X, Y - Rc))
        ahead = (along > 0.0) & (along < self.lookahead + self.front_offset) & (np.abs(lateral) < self.half_width)
        if not ahead.any():
            return math.inf
        return max(0.0, float(along[ahead].min()) - self.front_offset)

    # ----------------------------------------------------------------- control
    def control(self, now):
        dt = 0.0 if self.prev_t is None else now - self.prev_t
        self.prev_t = now
        v_set = min(self.max_speed, max(0.0, self.desired.linear.x))
        v_ego = self.v_cmd if self.v_ego is None else self.v_ego
        kappa, e_lat, e_head = self.lane_shape(now)
        gap = self.gap_ahead(kappa, e_lat, e_head)

        # Lead-vehicle speed = our speed + rate the gap is changing (smoothed).
        if math.isfinite(gap) and self.prev_gap is not None and math.isfinite(self.prev_gap) and dt > 1e-3:
            raw = v_ego + (gap - self.prev_gap) / dt
            self.lead_speed = raw if self.lead_speed is None else (1 - self.alpha) * self.lead_speed + self.alpha * raw
        elif not math.isfinite(gap):
            self.lead_speed = None
        if math.isfinite(gap) and (self.prev_gap is None or not math.isfinite(self.prev_gap)):
            self.get_logger().info(f'ROAD: target detected {gap:.2f} m ahead (corridor curvature {kappa:+.3f} 1/m)')
        self.prev_gap = gap

        if not math.isfinite(gap):
            v_target, state = v_set, 'CRUISE'
        elif gap < self.stop_distance:
            v_target, state = 0.0, 'STOP'
        else:
            lead = 0.0 if self.lead_speed is None else max(0.0, self.lead_speed)
            desired_gap = self.d0 + self.h * v_ego
            v_target = float(np.clip(lead + self.k_gap * (gap - desired_gap), 0.0, v_set))
            if self.state == 'FOLLOW':
                state = 'FOLLOW' if v_target < v_set - 0.01 else 'CRUISE'
            else:
                state = 'FOLLOW' if v_target < v_set - 0.05 else 'CRUISE'
            # hold a real stop (no creeping) until the way ahead clearly opens up
            if v_target < (0.08 if self.state == 'STOP' else 0.03):
                v_target, state = 0.0, 'STOP'

        # RSS proper response: cap the speed so every RSS safe distance is kept
        if self.rss_limit is not None and now - self.rss_t < 0.5 and v_target > self.rss_limit:
            v_target = self.rss_limit
            if state != 'STOP':
                state = 'RSS_LIMIT'

        # Comfort and actuator limits.
        if dt > 0.0:
            dv = float(np.clip(v_target - self.v_cmd, -self.max_decel * dt, self.max_accel * dt))
            self.v_cmd = max(0.0, self.v_cmd + dv)
        if state == 'STOP' and gap < self.stop_distance:
            self.v_cmd = 0.0  # emergency: do not wait for the decel limit

        # Keep the CNN's path curvature, so the car never spins on the spot when slowing.
        w = self.desired.angular.z * (self.v_cmd / v_set) if v_set > 1e-3 else 0.0

        cmd = Twist()
        cmd.linear.x = self.v_cmd
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        if state != self.state:
            self.get_logger().info(f'ROAD: {self.state} -> {state} (gap {gap:.2f} m)')
            self.state = state
        self.gap_pub.publish(Float32(data=float(gap) if math.isfinite(gap) else -1.0))
        self.lead_pub.publish(Float32(data=-1.0 if self.lead_speed is None else float(self.lead_speed)))
        self.state_pub.publish(String(data=state))

        self.count += 1
        if self.count % 60 == 0:
            lead = 'none' if self.lead_speed is None else f'{self.lead_speed:.2f}'
            self.get_logger().info(
                f'ROAD: {state} v={self.v_cmd:.2f} m/s gap={gap:.2f} m lead_speed={lead} m/s '
                f'corridor curvature={kappa:+.3f} 1/m')


def main(args=None):
    rclpy.init(args=args)
    node = RoadDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
