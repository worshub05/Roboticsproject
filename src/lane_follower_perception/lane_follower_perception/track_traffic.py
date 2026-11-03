#!/usr/bin/env python3
"""
Traffic on the oval track.

Scenarios (parameter 'scenario'):
  follow    one slow car ahead in our lane (ACC test on a bend)
  overtake  four cars: a slow car and a parked car in our lane, and two oncoming cars in the
            other lane (two-way road), for the overtaking test

Cars are spawned with /spawn_entity and moved along the lanes with /gazebo/set_entity_state.
Ground truth from Gazebo positions (not the LiDAR) is used to report the true gap to the car
ahead, the closest approach to any car, and collisions.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

from .collect_track_data import L, R, track_path

PER = 2 * L + 2 * math.pi * R
CAR_HALF_LENGTH, CAR_HALF_WIDTH = 0.25, 0.125
ROBOT_RADIUS = 0.17          # robot footprint is about 0.30 x 0.35 m
ROBOT_FRONT = 0.15
BLUE, RED = '0.1 0.3 0.8 1', '0.8 0.15 0.1 1'


def car_sdf(name, colour):
    return f"""<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><link name="body">
  <kinematic>true</kinematic><gravity>false</gravity>
  <inertial><mass>5.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
    <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
  <collision name="collision"><pose>0 0 0.225 0 0 0</pose>
    <geometry><box><size>0.5 0.25 0.35</size></box></geometry></collision>
  <visual name="body"><pose>0 0 0.18 0 0 0</pose><geometry><box><size>0.5 0.25 0.26</size></box></geometry>
    <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material></visual>
  <visual name="cabin"><pose>-0.03 0 0.36 0 0 0</pose><geometry><box><size>0.28 0.22 0.1</size></box></geometry>
    <material><ambient>0.15 0.15 0.2 1</ambient><diffuse>0.15 0.15 0.2 1</diffuse></material></visual>
</link></model></sdf>"""


class TrackTraffic(Node):
    def __init__(self):
        super().__init__('track_traffic')
        self.declare_parameter('scenario', 'follow')
        self.declare_parameter('lead_start_s', 24.0)     # follow scenario
        self.declare_parameter('lead_speed', 0.15)
        self.declare_parameter('slow_start_s', 8.0)      # overtake scenario
        self.declare_parameter('slow_speed', 0.1)
        self.declare_parameter('parked_s', 46.0)
        self.declare_parameter('oncoming_speed', 0.3)
        self.declare_parameter('oncoming_start_s', [21.0, 3.5])
        self.declare_parameter('rate', 20.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.scenario = g('scenario')

        # each car: name, lane offset (0 = our lane, 0.5 = other lane), s, speed along our s, colour
        if self.scenario == 'overtake':
            self.cars = [['slow_car', 0.0, g('slow_start_s'), g('slow_speed'), BLUE],
                         ['parked_car', 0.0, g('parked_s'), 0.0, BLUE]]
            for k, s0 in enumerate(g('oncoming_start_s')):
                self.cars.append([f'oncoming_car_{k + 1}', 0.5, float(s0), -g('oncoming_speed'), RED])
        elif self.scenario == 'cutin':
            # one car for sudden events, parked 3 m off the road between events
            self.cars = [['cutin_car', -3.0, 0.0, 0.0, RED]]
        elif self.scenario == 'lights':
            self.cars = []                     # traffic light test: no other cars
        else:
            self.cars = [['track_lead_car', 0.0, g('lead_start_s'), g('lead_speed'), BLUE]]

        ss = np.arange(0.0, PER, 0.02)
        self.ss = ss
        self.path = np.array([track_path(si, 0.0) for si in ss])
        self.robot = None            # x, y, s, speed
        self.spawned = False
        self.spawn_futures = []
        self.t_prev = None
        self.min_gap = math.inf      # true gap to the car ahead in our lane
        self.min_clear = math.inf    # closest approach to any car
        self.collided = set()
        self.count = 0
        # cut-in scenario: alternating events every 20 s (cut-in from the other lane / car popping up)
        self.declare_parameter('event_period', 20.0)
        self.declare_parameter('popup_gap', 0.4)        # m, bumper gap at which a hidden car appears
        self.declare_parameter('cutin_gap', 0.45)       # m, bumper gap at which the other car cuts in
        self.event = dict(phase='IDLE', n=0, t=None, t_next=None, min_clear=math.inf, kind='', info='')
        self.car_clear = {}
        self.event_results = []

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.gap_pub = self.create_publisher(Float32, '/traffic/true_gap', 10)
        self.cars_pub = self.create_publisher(Float32MultiArray, '/traffic/cars', 10)   # x, y, yaw per car
        self.cars_v_pub = self.create_publisher(Float32MultiArray, '/traffic/cars_v', 10)  # x, y, yaw, vx, vy
        self.prev_xy = {}
        self.create_timer(1.0 / g('rate'), self.tick)
        self.get_logger().info(f'Track traffic: scenario "{self.scenario}", cars ' +
                               ', '.join(f'{c[0]} (s={c[2]:.1f}, {c[3]:+.2f} m/s)' for c in self.cars))

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        i = int(np.argmin((self.path[:, 0] - x) ** 2 + (self.path[:, 1] - y) ** 2))
        px, py, th = self.path[i]
        d = -math.sin(th) * (x - px) + math.cos(th) * (y - py)        # m left of our lane centre
        self.robot = (x, y, float(self.ss[i]), msg.twist.twist.linear.x, d)

    def pose_of(self, car):
        _, off, s, v, _ = car
        x, y, th = track_path(s, off)
        return x, y, th + (math.pi if v < 0 else 0.0)

    def spawn(self):
        for car in self.cars:
            x, y, th = self.pose_of(car)
            req = SpawnEntity.Request()
            req.name = car[0]
            req.xml = car_sdf(car[0], car[4])
            req.initial_pose.position.x, req.initial_pose.position.y = float(x), float(y)
            req.initial_pose.orientation.z, req.initial_pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            req.reference_frame = 'world'
            self.spawn_futures.append(self.spawn_client.call_async(req))
        self.spawned = True
        self.get_logger().info(f'Track traffic: {len(self.cars)} cars spawned')

    def move(self, car):
        x, y, th = self.pose_of(car)
        req = SetEntityState.Request()
        req.state.name = car[0]
        req.state.pose.position.x, req.state.pose.position.y = float(x), float(y)
        req.state.pose.orientation.z, req.state.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        req.state.reference_frame = 'world'
        self.state_client.call_async(req)

    def tick(self):
        if not self.spawned:
            if self.spawn_client.service_is_ready():
                self.spawn()
            return
        if not self.state_client.service_is_ready() or not all(f.done() for f in self.spawn_futures):
            return              # wait until Gazebo has really created every car
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.t_prev is None else now - self.t_prev
        self.t_prev = now
        if self.scenario == 'cutin' and self.robot is not None:
            self.cutin_update(now, dt)
        truth = []
        for car in self.cars:
            car[2] = (car[2] + car[3] * dt) % PER
            self.move(car)
            truth += [float(v) for v in self.pose_of(car)]
        self.cars_pub.publish(Float32MultiArray(data=truth))
        truth_v = []
        for car in self.cars:
            x, y, th = self.pose_of(car)
            # exact velocity: speed along the lane (a lane offset to the inside is slower on a bend)
            sm = car[2] % PER
            bend = L <= sm < L + math.pi * R or sm >= 2 * L + math.pi * R
            speed = abs(car[3]) * ((R - car[1]) / R if bend else 1.0)
            truth_v += [float(x), float(y), float(th), speed * math.cos(th), speed * math.sin(th)]
        self.cars_v_pub.publish(Float32MultiArray(data=truth_v))
        if self.robot is None:
            return

        rx, ry, rs, rv, rd = self.robot
        gaps, clear = [], math.inf
        for car in self.cars:
            x, y, th = self.pose_of(car)
            c, s = math.cos(-th), math.sin(-th)
            lx, ly = (rx - x) * c - (ry - y) * s, (rx - x) * s + (ry - y) * c   # robot in the car's frame
            cl = max(abs(lx) - CAR_HALF_LENGTH - ROBOT_RADIUS, abs(ly) - CAR_HALF_WIDTH - ROBOT_RADIUS)
            clear = min(clear, cl)
            self.car_clear[car[0]] = cl
            if cl < 0.0 and car[0] not in self.collided:
                self.collided.add(car[0])
                self.get_logger().info(f'TRAFFIC: ❌ COLLISION with {car[0]}')
            if car[1] == 0.0:
                gaps.append((car[2] - rs) % PER - CAR_HALF_LENGTH - ROBOT_FRONT)
        self.min_clear = min(self.min_clear, clear)
        gap = min(gaps) if gaps else math.inf
        if gap < PER / 2 and abs(rd) < 0.2:          # only while we are in our own lane (not overtaking)
            self.gap_pub.publish(Float32(data=float(gap)))
            self.min_gap = min(self.min_gap, gap)
        self.count += 1
        if self.count % 80 == 0:
            self.get_logger().info(f'TRAFFIC: robot at s={rs:.1f} m, {rv:.2f} m/s, true gap to car ahead in our lane '
                                   f'{gap:.2f} m, closest approach to any car so far {self.min_clear:.2f} m')


def _cutin_update(self, now, dt):
    """Sudden events for the emergency braking test (scenario 'cutin')."""
    ev, car = self.event, self.cars[0]
    rx, ry, rs, rv, rd = self.robot
    gap = (car[2] - rs) % PER - CAR_HALF_LENGTH - ROBOT_FRONT        # bumper gap along our lane
    if ev['phase'] == 'IDLE':
        car[1], car[2], car[3] = -3.0, rs, 0.0                        # wait off the road, beside us
        if ev['t_next'] is None:
            ev['t_next'] = now + 15.0
        if now >= ev['t_next'] and rv > 0.3 and abs(rd) < 0.1:
            ev['n'] += 1
            ev['min_clear'] = math.inf
            ev['t'] = now
            if ev['n'] % 2 == 1:                                      # cut-in from the other lane
                ev['kind'], ev['phase'] = 'cut-in', 'APPROACH'
                car[1], car[2], car[3] = 0.5, (rs + 1.2 + CAR_HALF_LENGTH + ROBOT_FRONT) % PER, 0.2
            else:                                                     # hidden stopped car pops up
                g = self.get_parameter('popup_gap').value
                ev['kind'], ev['phase'] = 'pop-up', 'STOPPED'
                car[1], car[2], car[3] = 0.0, (rs + g + CAR_HALF_LENGTH + ROBOT_FRONT) % PER, 0.0
                ev['info'] = f'stopped car appeared {g:.2f} m ahead while we drove {rv:.2f} m/s'
    elif ev['phase'] == 'APPROACH':                                   # in the other lane, we close in
        if gap < self.get_parameter('cutin_gap').value or now - ev['t'] > 8.0:
            ev['phase'], ev['t_cut'] = 'CUT', now
            ev['info'] = f'car cut in {gap:.2f} m ahead while we drove {rv:.2f} m/s, then braked hard'
    elif ev['phase'] == 'CUT':                                        # 0.5 m sideways in 0.5 s
        car[1] = max(0.0, car[1] - 1.0 * dt)
        if car[1] <= 0.15:
            car[3] = max(0.0, car[3] - 1.5 * dt)                     # brakes hard once in our lane
        if car[1] == 0.0 and car[3] == 0.0:
            ev['phase'], ev['t'] = 'STOPPED', now
    if ev['phase'] in ('CUT', 'STOPPED'):
        ev['min_clear'] = min(ev['min_clear'], self.car_clear.get(car[0], math.inf))
        if ev['phase'] == 'STOPPED' and now - ev['t'] > 4.0:
            ok = ev['min_clear'] > 0.0
            self.event_results.append(ok)
            self.get_logger().info(f'TRAFFIC: {ev["kind"]} {ev["n"]}: {ev["info"]}; closest approach '
                                   f'{max(0.0, ev["min_clear"]):.2f} m, {"✅ no collision" if ok else "❌ collision"} '
                                   f'(events so far {len(self.event_results)}, collisions '
                                   f'{self.event_results.count(False)})')
            ev['phase'], ev['t_next'] = 'IDLE', now + self.get_parameter('event_period').value


TrackTraffic.cutin_update = _cutin_update


def main(args=None):
    rclpy.init(args=args)
    node = TrackTraffic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        print(f'TRAFFIC SUMMARY: closest approach to any car {node.min_clear:.2f} m, '
              f'smallest gap to a car ahead {node.min_gap:.2f} m, '
              f'collisions: {", ".join(sorted(node.collided)) if node.collided else "none"}', flush=True)
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
