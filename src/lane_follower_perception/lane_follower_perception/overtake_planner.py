#!/usr/bin/env python3
"""
Behaviour planner: overtake a slow or parked car on a two-way road.

Sits between the CNN and the road driver:

    multitask CNN --/cnn/cmd_vel, /cnn/lane/*--> [overtake_planner] --/cmd_vel_desired, /lane/*--> road_driver

States
  LANE_KEEP     CNN drives (its steering and lane estimates are passed straight through).
  PEEK          a slow car is close ahead: move 0.1 m towards the centre line so the LiDAR can
                see past it, keep following, and wait until the other lane is clear.
  CHANGE_LEFT   smooth lane change into the other lane (quintic profile).
  PASS          drive in the other lane at passing speed until our lane is clear beside us.
  CHANGE_RIGHT  smooth lane change back; the CNN takes over again once we are centred.
  ABORT         an oncoming car appeared before we drew level: go back behind the slow car.

Gap acceptance (occlusion aware): the other lane is only "clear" if every point on its
centre line, up to the distance the manoeuvre needs, is SEEN to be empty by the LiDAR. A
point hidden behind the slow car counts as not clear. The distance needed is
    D = (v_pass + v_oncoming_max) * T_overtake + margin,
so an oncoming car outside the check zone cannot reach us before we are back.

While overtaking, steering comes from a path tracker that uses odometry and the known track
layout (a map), because the CNN was only trained in our own lane.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String

from .collect_track_data import L, R, track_path

PER = 2 * L + 2 * math.pi * R


def to_frenet(x, y):
    """Track coordinates for points (arrays): s along our lane centre, d to the left of it."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.empty_like(x)
    d = np.empty_like(x)
    bottom = (x >= 0) & (x <= L) & (y < R)
    top = (x >= 0) & (x <= L) & (y >= R)
    right = x > L
    left = x < 0
    s[bottom], d[bottom] = x[bottom], y[bottom]
    s[top], d[top] = L + math.pi * R + (L - x[top]), 2 * R - y[top]
    a = np.arctan2(x[right] - L, -(y[right] - R))
    s[right], d[right] = L + R * a, R - np.hypot(x[right] - L, y[right] - R)
    a = np.arctan2(-x[left], y[left] - R)
    s[left], d[left] = 2 * L + math.pi * R + R * a, R - np.hypot(x[left], y[left] - R)
    return s % PER, d


def lane_curvature(s):
    s = s % PER
    on_straight = s < L or (L + math.pi * R <= s < 2 * L + math.pi * R)
    return 0.0 if on_straight else 1.0 / R


def ahead(s_from, s_to):
    """Distance along the track from s_from forward to s_to (0..PER)."""
    return (s_to - s_from) % PER


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u), 30 * u ** 2 * (1 - u) ** 2


class OvertakePlanner(Node):
    def __init__(self):
        super().__init__('overtake_planner')
        p = self.declare_parameter
        p('cruise_speed', 0.5)          # m/s in our lane
        p('pass_speed', 0.7)            # m/s while overtaking
        p('oncoming_max_speed', 0.35)   # m/s, assumed worst case for oncoming traffic
        p('lane_width', 0.5)            # m, other lane centre is this far to the left
        p('peek_offset', 0.1)           # m to the left while waiting to see past the slow car
        p('change_length', 1.2)         # m travelled during one lane change
        p('peek_length', 0.6)           # m travelled while edging over to peek
        p('peek_gap', 1.1)              # m, hang back this far behind the slow car (room to pull out)
        p('trigger_gap', 3.5)           # m, consider overtaking when the slow car is this close
        p('slow_ratio', 0.6)            # lead speed below this x cruise speed counts as slow
        p('trigger_time', 0.5)          # s the slow car must stay slow and close
        p('clear_time', 0.5)            # s the other lane must look clear before going
        p('return_gap', 0.6)            # m our lane must be clear behind our centre before returning
        p('margin', 1.0)                # m extra in the clear-distance calculation
        p('max_check', 9.5)             # m, LiDAR max range is 10 m: cannot confirm further
        p('robot_length', 0.35)
        p('car_length', 0.5)
        p('gain_lateral', 1.5)
        p('gain_heading', 2.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.c = {n: g(n) for n in ('cruise_speed', 'pass_speed', 'oncoming_max_speed', 'lane_width',
                                    'peek_offset', 'change_length', 'peek_length', 'peek_gap', 'trigger_gap', 'slow_ratio',
                                    'trigger_time', 'clear_time', 'return_gap', 'margin', 'max_check',
                                    'robot_length', 'car_length', 'gain_lateral', 'gain_heading')}

        self.state = 'LANE_KEEP'
        self.pose = None                 # x, y, yaw, v
        self.fr = None                   # s, d, heading error
        self.scan = None
        self.cnn_cmd = Twist()
        self.cnn_lane = {'curvature': 0.0, 'offset': 0.0, 'heading': 0.0}
        self.road_gap, self.lead_speed = math.inf, None
        self.slow_since = None
        self.clear_since = None
        self.change = None               # (s_start, d_from, d_to, length) of the current lateral move
        self.d_hold = 0.0                # lateral target when not in a lane change
        self.last_log = 0.0
        self.go = None                   # (time, clear distance) when the overtake was started
        self.stopped_since = None
        self.overtakes = 0

        self.create_subscription(Twist, '/cnn/cmd_vel', self.cnn_cmd_cb, 10)
        for n in self.cnn_lane:
            self.create_subscription(Float32, '/cnn/lane/' + n, lambda m, k=n: self.cnn_lane.__setitem__(k, m.data), 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Float32, '/road/gap', lambda m: setattr(self, 'road_gap', m.data if m.data >= 0 else math.inf), 10)
        self.create_subscription(Float32, '/road/lead_speed', lambda m: setattr(self, 'lead_speed', None if m.data < 0 else m.data), 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)
        self.lane_pubs = {n: self.create_publisher(Float32, '/lane/' + n, 10) for n in self.cnn_lane}
        self.state_pub = self.create_publisher(String, '/behaviour/state', 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info('Overtake planner ready (two-way road, occlusion-aware gap check)')

    # ------------------------------------------------------------------ inputs
    def cnn_cmd_cb(self, msg):
        self.cnn_cmd = msg

    def scan_cb(self, msg):
        self.scan = msg

    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw, msg.twist.twist.linear.x)
        s, d = to_frenet([p.x], [p.y])
        th = track_path(float(s[0]), 0.0)[2]
        self.fr = (float(s[0]), float(d[0]), wrap(yaw - th))

    # -------------------------------------------------------------- perception
    def scan_points(self):
        """LiDAR returns in world coordinates."""
        sc = self.scan
        r = np.asarray(sc.ranges, float)
        ang = sc.angle_min + np.arange(len(r)) * sc.angle_increment
        ok = np.isfinite(r) & (r > sc.range_min)
        x0, y0, yaw, _ = self.pose
        return x0 + r[ok] * np.cos(ang[ok] + yaw), y0 + r[ok] * np.sin(ang[ok] + yaw)

    def lane_occupied(self, d_lo, d_hi, s_back, s_fwd):
        """Any LiDAR return with d in [d_lo, d_hi] from s_back behind to s_fwd ahead of us?"""
        if self.scan is None:
            return True
        px, py = self.scan_points()
        if px.size == 0:
            return False
        s, d = to_frenet(px, py)
        rel = (s - self.fr[0] + PER / 2) % PER - PER / 2
        return bool(np.any((d > d_lo) & (d < d_hi) & (rel > -s_back) & (rel < s_fwd)))

    def other_lane_clear(self, dist):
        """
        Occlusion-aware check of the other lane's centre line from just behind us to 'dist' ahead.
        Returns (clear, reason).
        """
        if dist > self.c['max_check']:
            return False, f'need {dist:.1f} m clear but the LiDAR only reaches {self.c["max_check"]:.1f} m'
        sc = self.scan
        r = np.asarray(sc.ranges, float)
        n = len(r)
        x0, y0, yaw, _ = self.pose
        s0 = self.fr[0]
        w = self.c['lane_width']
        for k in np.arange(-0.75, dist + 1e-6, 0.25):   # from just behind us (a car alongside counts)
            px, py, _ = track_path(s0 + k, w)
            dx, dy = px - x0, py - y0
            rng = math.hypot(dx, dy)
            if rng < 0.3:
                continue
            i = int(round((wrap(math.atan2(dy, dx) - yaw) - sc.angle_min) / sc.angle_increment)) % n
            beams = r[[(i - 1) % n, i, (i + 1) % n]]
            m = float(np.where(np.isfinite(beams), beams, 99.0).min())   # closest return near this bearing
            if abs(m - rng) <= 0.35:
                return False, f'oncoming car {k:.1f} m ahead in the other lane'
            if m < rng - 0.35:
                return False, f'cannot see the other lane {k:.1f} m ahead (hidden behind the slow car)'
        return True, f'other lane seen clear for {dist:.1f} m'

    def needed_distance(self):
        """How far ahead the other lane must be clear for the whole overtake."""
        c = self.c
        v_lead = 0.0 if self.lead_speed is None else max(0.0, self.lead_speed)
        v_rel = max(0.15, c['pass_speed'] - v_lead)
        gap = self.road_gap if math.isfinite(self.road_gap) else c['trigger_gap']
        to_pass = gap + c['car_length'] + c['robot_length'] + c['return_gap']
        t = to_pass / v_rel + 2 * c['change_length'] / c['pass_speed']
        return (c['pass_speed'] + c['oncoming_max_speed']) * t + c['margin']

    # ----------------------------------------------------------------- control
    def d_reference(self):
        """Lateral target (m left of our lane centre) and its slope dd/ds."""
        if self.change is None:
            return self.d_hold, 0.0
        s_start, d_from, d_to, length = self.change
        u = ahead(s_start, self.fr[0]) / length
        f, df = smoothstep(u)
        return d_from + (d_to - d_from) * f, (d_to - d_from) * df / length

    def start_change(self, d_to, length=None):
        self.change = (self.fr[0], self.fr[1], d_to, length or self.c['change_length'])
        self.d_hold = d_to

    def set_state(self, new, why):
        if new != self.state:
            self.get_logger().info(f'BEHAVIOUR: {self.state} -> {new} ({why})')
            self.state = new

    def tick(self):
        if self.pose is None or self.scan is None:
            return
        c = self.c
        now = self.get_clock().now().nanoseconds * 1e-9
        s, d, e_head = self.fr
        st = self.state

        lead_close = math.isfinite(self.road_gap) and self.road_gap < c['trigger_gap']
        lead_slow = self.lead_speed is not None and self.lead_speed < c['slow_ratio'] * c['cruise_speed']

        if st == 'LANE_KEEP':
            if lead_close and lead_slow:
                self.slow_since = now if self.slow_since is None else self.slow_since
                if now - self.slow_since > c['trigger_time']:
                    self.start_change(c['peek_offset'], c['peek_length'])
                    self.clear_since = None
                    self.set_state('PEEK', f'slow car {self.road_gap:.1f} m ahead at {self.lead_speed:.2f} m/s')
            else:
                self.slow_since = None

        elif st == 'PEEK':
            if not (math.isfinite(self.road_gap) and self.road_gap < c['trigger_gap'] + 1.0):
                self.start_change(0.0)
                self.set_state('CHANGE_RIGHT', 'slow car no longer ahead')
            else:
                need = self.needed_distance()
                ok, why = self.other_lane_clear(need)
                if ok:
                    self.clear_since = now if self.clear_since is None else self.clear_since
                    if now - self.clear_since > c['clear_time']:
                        self.start_change(c['lane_width'])
                        self.go = (now, need)
                        self.set_state('CHANGE_LEFT', why)
                else:
                    self.clear_since = None
                    if now - self.last_log > 2.0:
                        self.get_logger().info(f'BEHAVIOUR: waiting behind slow car: {why}')
                        self.last_log = now

        elif st in ('CHANGE_LEFT', 'PASS'):
            # committed once the slow car is level with us: going back is then no longer possible
            committed = self.lane_occupied(-0.3, 0.3, 1.0, 0.6)
            # Anything outside the distance checked at the start cannot reach us in time if it keeps
            # to the assumed speed limit, so only the part of that distance still needed is re-checked.
            # A car inside it means it is faster than assumed: go back while we still can.
            remaining = self.go[1] - (c['pass_speed'] + c['oncoming_max_speed']) * (now - self.go[0])
            if not committed and remaining > 0.5:
                ok, why = self.other_lane_clear(remaining)
                if not ok and 'oncoming' in why:
                    self.start_change(0.0)
                    self.set_state('ABORT', why)
                    return self.publish(now)
            if st == 'CHANGE_LEFT' and ahead(self.change[0], s) >= self.change[3]:
                self.set_state('PASS', 'in the other lane')
            elif self.state == 'PASS' and not self.lane_occupied(-0.3, 0.3, c['return_gap'], 1.5):
                self.start_change(0.0)
                self.set_state('CHANGE_RIGHT', 'passed the slow car, our lane is clear')

        elif st == 'ABORT' and self.pose[3] < 0.02:
            # stuck behind a stopped car part-way out: wait, then try again when the lane is clear
            self.stopped_since = now if self.stopped_since is None else self.stopped_since
            if now - self.stopped_since > 2.0:
                need = self.needed_distance()
                ok, why = self.other_lane_clear(need)
                if ok:
                    self.start_change(c['lane_width'])
                    self.go, self.stopped_since = (now, need), None
                    self.set_state('CHANGE_LEFT', 'retry: ' + why)
        elif st in ('CHANGE_RIGHT', 'ABORT'):
            self.stopped_since = None
            if ahead(self.change[0], s) >= self.change[3] and abs(d) < 0.06 and abs(e_head) < 0.12:
                if st == 'CHANGE_RIGHT' and self.d_hold == 0.0 and self.change[1] > 0.3:
                    self.overtakes += 1
                    self.get_logger().info(f'BEHAVIOUR: ✅ overtake {self.overtakes} complete')
                self.change = None
                self.slow_since = None
                self.set_state('LANE_KEEP', 'back in our lane, CNN driving')
        self.publish(now)

    def publish(self, now):
        c = self.c
        s, d, e_head = self.fr
        st = self.state
        self.state_pub.publish(String(data=st))
        if st == 'LANE_KEEP':
            cmd = Twist()
            cmd.linear.x = min(self.cnn_cmd.linear.x, c['cruise_speed'])
            cmd.angular.z = self.cnn_cmd.angular.z
            self.cmd_pub.publish(cmd)
            for n, v in self.cnn_lane.items():
                self.lane_pubs[n].publish(Float32(data=float(v)))
            return

        # map-based path tracking towards the lateral reference
        d_ref, slope = self.d_reference()
        kappa = lane_curvature(s)
        k_path = kappa / max(0.2, 1.0 - kappa * d_ref)
        v = c['pass_speed'] if st == 'PASS' else c['cruise_speed']
        if st == 'PEEK' and math.isfinite(self.road_gap):
            # hang back a little further than normal following, so there is room to pull out
            lead = 0.0 if self.lead_speed is None else max(0.0, self.lead_speed)
            v = min(v, max(0.0, lead + 0.6 * (self.road_gap - c['peek_gap'])))
            # a car can only move sideways by driving forward: creep until the peek position is reached
            if abs(d - c['peek_offset']) > 0.03 and self.road_gap > 0.75:
                v = max(v, 0.08)
        w = v * k_path - c['gain_lateral'] * (d - d_ref) - c['gain_heading'] * (e_head - math.atan(slope))
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = float(np.clip(w, -1.5, 1.5))
        self.cmd_pub.publish(cmd)

        # the ACC corridor: the other lane while moving into it or passing, otherwise our own lane
        centre = c['lane_width'] if st in ('CHANGE_LEFT', 'PASS') else 0.0
        k_ahead = np.mean([lane_curvature(s + a) for a in (0.5, 1.0, 1.5, 2.0, 2.5)])
        self.lane_pubs['curvature'].publish(Float32(data=float(k_ahead / max(0.2, 1.0 - k_ahead * centre))))
        self.lane_pubs['offset'].publish(Float32(data=float(d - centre)))
        self.lane_pubs['heading'].publish(Float32(data=float(e_head)))


def main(args=None):
    rclpy.init(args=args)
    node = OvertakePlanner()
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
