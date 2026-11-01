#!/usr/bin/env python3
"""
RSS safety monitor.

Every time the LiDAR tracker publishes /tracks it checks our robot against every tracked car with
the RSS rules (rss.py): a car is DANGEROUS when it is too close both along the road and sideways.

Publishes
  /safety/speed_limit  Float32  highest speed that keeps every RSS distance (-1 = no limit).
                                The road driver never drives faster than this (the RSS
                                "proper response": brake when a situation becomes dangerous).
  /safety/rss          String   SAFE or DANGER
and logs every dangerous situation plus running statistics ("RSS STATS: ..."), so the planner's
decisions (peek, lane change, pass, return) can all be checked afterwards.

Parameter enforce:=false only monitors (publishes -1), for a with / without comparison.
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, String

from .predict import lane_of
from .rss import RSSParams, check, relative, to_track
from .track_map import LANE_WIDTH


class RSSMonitor(Node):
    def __init__(self):
        super().__init__('rss_monitor')
        self.declare_parameter('enforce', True)
        defaults = RSSParams()
        for k, v in vars(defaults).items():
            self.declare_parameter(k, v)
        self.p = RSSParams(**{k: float(self.get_parameter(k).value) for k in vars(defaults)})
        self.enforce = bool(self.get_parameter('enforce').value)
        self.pose = None                      # x, y, yaw, v
        self.t_prev = None
        self.in_danger, self.danger_start, self.safe_since = False, None, None
        self.stats = dict(t=0.0, t_danger=0.0, n_danger=0, t_limited=0.0, min_margin=math.inf)
        self.count = 0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32MultiArray, '/tracks', self.tracks_callback, 10)
        self.limit_pub = self.create_publisher(Float32, '/safety/speed_limit', 10)
        self.state_pub = self.create_publisher(String, '/safety/rss', 10)
        p = self.p
        self.get_logger().info(f'RSS monitor ready ({"enforcing" if self.enforce else "monitoring only"}): '
                               f'rho {p.rho} s, a_acc {p.a_acc}, b_min {p.b_min}, b_max {p.b_max}, '
                               f'b_correct {p.b_correct} m/s2, lateral a {p.a_lat} / b {p.b_lat} m/s2, mu {p.mu} m')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw, msg.twist.twist.linear.x)

    def tracks_callback(self, msg):
        if self.pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.t_prev is None else min(0.5, max(0.0, now - self.t_prev))
        self.t_prev = now
        x, y, yaw, v = self.pose
        ego_t = to_track(x, y, yaw, v)
        ego = dict(v_lon=max(0.0, ego_t[2]), v_lat=ego_t[3])
        ego_wrong = ego_t[1] > LANE_WIDTH / 2

        d = msg.data
        limit, worst, danger = math.inf, None, False
        for i in range(0, len(d) - 7, 8):
            tid, cx, cy, vx, vy, _, length, width = d[i:i + 8]
            car_t = to_track(cx, cy, vx=vx, vy=vy)
            car = relative(ego_t, car_t, max(0.25, length / 2), max(0.125, width / 2))
            car_wrong = car['v_lon'] < -0.02 and lane_of(car['d']) == 0
            r = check(ego, car, self.p, ego_wrong, car_wrong)
            if not r['relevant']:
                continue
            if r['lat_gap'] < r['lat_safe']:
                self.stats['min_margin'] = min(self.stats['min_margin'], r['lon_gap'] - r['lon_safe'])
            if r['dangerous']:
                danger = True
                if worst is None or r['lon_gap'] - r['lon_safe'] < worst[1]['lon_gap'] - worst[1]['lon_safe']:
                    worst = (int(tid), r)
            limit = min(limit, r['v_limit'])

        # statistics
        st = self.stats
        st['t'] += dt
        if danger:
            st['t_danger'] += dt
        if limit < ego['v_lon']:
            st['t_limited'] += dt
        if danger and not self.in_danger:
            self.in_danger, self.danger_start = True, now
            st['n_danger'] += 1
            tid, r = worst
            self.get_logger().info(
                f'RSS: ⚠ DANGEROUS situation with car #{tid} ({r["case"]} direction): gap along the road '
                f'{r["lon_gap"]:.2f} m < safe {r["lon_safe"]:.2f} m and sideways {r["lat_gap"]:.2f} m < '
                f'{r["lat_safe"]:.2f} m; our speed {ego["v_lon"]:.2f} m/s, limited to {min(limit, 9.9):.2f} m/s')
        if not danger and self.in_danger:
            self.safe_since = self.safe_since or now
            if now - self.safe_since > 0.3:
                self.get_logger().info(f'RSS: ✅ safe again after {now - self.danger_start:.1f} s')
                self.in_danger, self.safe_since = False, None
        elif danger:
            self.safe_since = None

        lim = limit if (self.enforce and math.isfinite(limit)) else -1.0
        self.limit_pub.publish(Float32(data=float(lim)))
        self.state_pub.publish(String(data='DANGER' if danger else 'SAFE'))

        self.count += 1
        if self.count % 50 == 0:
            pct = 100 * st['t_danger'] / max(st['t'], 1e-6)
            mm = f'{st["min_margin"]:.2f} m' if math.isfinite(st['min_margin']) else 'n/a'
            self.get_logger().info(
                f'RSS STATS: checked {st["t"]:.0f} s, dangerous situations {st["n_danger"]} '
                f'({st["t_danger"]:.1f} s, {pct:.1f}%), speed limited by RSS {st["t_limited"]:.1f} s, '
                f'smallest margin to a safe distance {mm}')


def main(args=None):
    rclpy.init(args=args)
    node = RSSMonitor()
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
