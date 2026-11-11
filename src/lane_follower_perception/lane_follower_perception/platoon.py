#!/usr/bin/env python3
"""
Platoon world: a leader car and a simulated follower in front of our robot, V2V broadcast and a
ground-truth referee.

  car 0  leader (blue): cruises at 0.45 m/s and every 25 s brakes to 0.15 m/s, waits, speeds up
  car 1  simulated follower (green): longitudinal model (comfort limits + 0.1 s lag) with the SAME
         law as our robot: mode acc (time gap 1.0 s), acc_short (0.3 s, no V2V) or cacc (0.3 s + V2V)
  robot  our real Gazebo car follows car 1 with the road driver (ACC) or cacc_controller (CACC)

Both cars are moved along our lane with /gazebo/set_entity_state. They start standing a little
ahead of the robot and drive off 3 s after they appear (or as soon as the robot is moving).

V2V: /v2v/platoon (std_msgs/String JSON: id, s, v, a for cars 0 and 1) at 10 Hz, with optional
loss (share of messages dropped) and delay (s).

Referee (Gazebo ground truth) per braking event: how far each car's speed dips (leader, car 1,
robot), amplification = robot dip / leader dip (> 1: disturbances grow = string unstable), the
smallest gaps, and collisions. Logs "PLATOON: ..." lines and "PLATOON STATS: ...".
"""
import math
import random

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from .cacc import ACCLaw, CACCLaw, LeaderProfile, parse_v2v, slew, v2v_message, v2v_stamp
from .track_map import PER, to_frenet, track_path
from .track_traffic import car_sdf

CAR_HALF = 0.25
ROBOT_FRONT, ROBOT_BACK = 0.15, 0.20
COLOURS = {0: '0.1 0.3 0.8 1', 1: '0.1 0.6 0.2 1'}
SPACING = {'acc': (0.5, 1.0), 'acc_short': (0.4, 0.3), 'cacc': (0.4, 0.3)}     # standstill gap, time gap


class Platoon(Node):
    def __init__(self):
        super().__init__('platoon')
        for k, v in dict(mode='cacc', rate=20.0, loss=0.0, delay=0.0, seed=1, cruise=0.45, v_low=0.15,
                         period=25.0, first=12.0, blackout=-1.0).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mode = g('mode')
        d0, h = SPACING.get(self.mode, SPACING['cacc'])
        self.d0, self.h = d0, h
        self.profile = LeaderProfile(cruise=float(g('cruise')), v_low=float(g('v_low')), period=float(g('period')),
                                     first=float(g('first')))
        self.law = CACCLaw(r=d0, h=h) if self.mode == 'cacc' else ACCLaw(d0=d0, h=h)
        self.loss, self.delay = float(g('loss')), float(g('delay'))
        self.blackout, self.blackout_logged = float(g('blackout')), False   # s after drive-off, < 0 = never
        self.rnd = random.Random(int(g('seed')))
        self.dt = 1.0 / float(g('rate'))
        self.cars = None                 # [{'id', 's', 'v', 'a', 'v_cmd'}] once placed
        self.robot = None                # front s, v
        self.t_go, self.t_prev, self.queue, self.heard = None, None, [], None   # heard: leader (v, a) via V2V
        self.vf = None                   # smoothed robot speed (odometry is noisy)
        self.spawned, self.spawn_futures = False, []
        self.event, self.ev = 0, None
        self.results, self.min_gap_robot, self.min_gap_car1, self.collisions = [], math.inf, math.inf, 0
        self.in_collision, self.next_scan, self.meas_gap, self.count = False, 0.0, None, 0
        self.pub = self.create_publisher(String, '/v2v/platoon', 10)
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f'Platoon: mode {self.mode} (standstill gap {d0} m, time gap {h} s), leader cruise '
                               f'{self.profile.cruise} m/s, braking to {self.profile.v_low} m/s every '
                               f'{self.profile.period:.0f} s; V2V loss {self.loss:.0%}, delay {self.delay} s')

    def odom_callback(self, msg):
        s, _ = to_frenet(msg.pose.pose.position.x, msg.pose.pose.position.y)
        v = msg.twist.twist.linear.x
        self.vf = v if self.vf is None else 0.7 * self.vf + 0.3 * v
        self.robot = ((float(s[0]) + ROBOT_FRONT) % PER, self.vf)

    # ---------------------------------------------------------------- Gazebo
    def pose_req(self, car):
        x, y, th = track_path(car['s'], 0.0)
        return x, y, th

    def spawn(self):
        for car in self.cars:
            x, y, th = self.pose_req(car)
            req = SpawnEntity.Request()
            req.name, req.reference_frame = f'platoon_car_{car["id"]}', 'world'
            req.xml = car_sdf(req.name, COLOURS[car['id']])
            req.initial_pose.position.x, req.initial_pose.position.y = float(x), float(y)
            req.initial_pose.orientation.z, req.initial_pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            self.spawn_futures.append(self.spawn_client.call_async(req))
        self.spawned = True

    def move(self, car):
        x, y, th = self.pose_req(car)
        req = SetEntityState.Request()
        req.state.name, req.state.reference_frame = f'platoon_car_{car["id"]}', 'world'
        req.state.pose.position.x, req.state.pose.position.y = float(x), float(y)
        req.state.pose.orientation.z, req.state.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        self.state_client.call_async(req)

    # ---------------------------------------------------------------- main loop
    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.robot is None:
            return
        if self.cars is None:                          # place the cars just ahead of the robot
            front = self.robot[0]
            gap = self.d0 + 0.3
            c1 = front + gap + CAR_HALF
            c0 = c1 + 2 * CAR_HALF + gap
            self.cars = [dict(id=0, s=c0 % PER, v=0.0, a=0.0, v_cmd=0.0), dict(id=1, s=c1 % PER, v=0.0, a=0.0, v_cmd=0.0)]
        if not self.spawned:
            if self.spawn_client.service_is_ready():
                self.spawn()
            return
        if not (self.state_client.service_is_ready() and all(f.done() for f in self.spawn_futures)):
            return
        dt = 0.0 if self.t_prev is None else min(0.2, now - self.t_prev)
        self.t_prev = now
        self.t_ready = getattr(self, 't_ready', None) or now
        lead, car1 = self.cars
        if self.t_go is None and (self.robot[1] > 0.25 or now - self.t_ready > 3.0):
            self.t_go = now
            self.get_logger().info('PLATOON: platoon drives off')
        # leader
        if self.t_go is not None:
            t = now - self.t_go
            vp, ap, ev = self.profile.speed(t)
            if t < 2.0:                                  # gentle start
                vp, ap = min(vp, 0.3 * t), (0.3 if 0.3 * t < vp else 0.0)
            lead['v'], lead['a'] = vp, ap
            if ev != self.event:
                self.start_event(ev, now)
        # V2V broadcast (with the simulated radio link)
        silent = self.blackout >= 0 and self.t_go is not None and now - self.t_go > self.blackout
        if silent and not self.blackout_logged:
            self.get_logger().info('PLATOON: ⚠ V2V blackout - no more messages')
            self.blackout_logged = True
        if not silent and self.rnd.random() >= self.loss:
            self.queue.append((now + self.delay, v2v_message(self.cars, now)))
        while self.queue and self.queue[0][0] <= now:
            msg = self.queue.pop(0)[1]
            self.pub.publish(String(data=msg))
            c0 = parse_v2v(msg).get(0)
            if c0 is not None:
                self.heard = (c0['v'], c0['a'], v2v_stamp(msg) or now)
        # car 1: same law as the robot
        gap1 = (lead['s'] - car1['s']) % PER - 2 * CAR_HALF
        if now >= self.next_scan:
            self.next_scan = now + 0.1
            self.meas_gap = gap1 + self.rnd.gauss(0.0, 0.01)
        if self.t_go is not None and dt > 0:
            if self.mode == 'cacc':
                age = max(0.0, now - self.heard[2]) if self.heard else 0.0
                target, car1['a'] = self.law.step(self.meas_gap, car1['v'], self.v2v_lead_v, self.v2v_lead_a, dt,
                                                  age=age)
            else:
                target = self.law.step(self.meas_gap, car1['v'], dt)
                car1['a'] = 0.0
            car1['v_cmd'] = slew(car1['v_cmd'], min(target, 0.5), dt)
            car1['v'] += (car1['v_cmd'] - car1['v']) * min(1.0, dt / 0.1)
        for car in self.cars:
            car['s'] = (car['s'] + car['v'] * dt) % PER
            self.move(car)
        self.referee(now, gap1)

    @property
    def v2v_lead_v(self):
        """What car 1 last heard from the leader over V2V (the same radio link as the robot)."""
        return self.heard[0] if self.heard else self.cars[0]['v']

    @property
    def v2v_lead_a(self):
        return self.heard[1] if self.heard else 0.0

    # ---------------------------------------------------------------- referee
    def start_event(self, ev, now):
        self.finish_event()
        self.event = ev
        self.ev = dict(n=ev, t=now, vmin=[math.inf, math.inf, math.inf], gmin=[math.inf, math.inf])

    def finish_event(self):
        e = self.ev
        if e is None or not math.isfinite(e['vmin'][2]):
            return
        cr = self.profile.cruise
        dips = [max(0.0, cr - x) for x in e['vmin']]
        amp = dips[2] / dips[0] if dips[0] > 0.01 else float('nan')
        self.results.append((dips, amp, e['gmin']))
        self.get_logger().info(f'PLATOON: braking event {e["n"]}: speed dips leader {dips[0]:.2f}, car 1 {dips[1]:.2f}, '
                               f'robot {dips[2]:.2f} m/s -> amplification {amp:.2f} '
                               f'{"✅" if amp <= 1.05 else "⚠ growing"}; smallest gaps car 1 {e["gmin"][0]:.2f} m, '
                               f'robot {e["gmin"][1]:.2f} m')
        self.ev = None

    def referee(self, now, gap1):
        lead, car1 = self.cars
        front, v_robot = self.robot
        gap_r = (car1['s'] - CAR_HALF - front) % PER
        if gap_r > PER / 2:
            gap_r -= PER
        if self.t_go is not None and now - self.t_go > 5.0:
            self.min_gap_robot = min(self.min_gap_robot, gap_r)
            self.min_gap_car1 = min(self.min_gap_car1, gap1)
        hit = gap_r < 0.02
        if hit and not self.in_collision:
            self.collisions += 1
            self.get_logger().info(f'PLATOON: ❌ COLLISION robot -> car 1 (gap {gap_r:.2f} m)')
        self.in_collision = hit
        e = self.ev
        if e is not None:
            for i, v in enumerate((lead['v'], car1['v'], v_robot)):
                e['vmin'][i] = min(e['vmin'][i], v)
            e['gmin'][0], e['gmin'][1] = min(e['gmin'][0], gap1), min(e['gmin'][1], gap_r)
            if now - e['t'] > 15.0:
                self.finish_event()
        self.count += 1
        if self.count % 100 == 0 and self.t_go is not None:
            amps = [r[1] for r in self.results if r[1] == r[1]]
            amp = f'{sum(amps) / len(amps):.2f}' if amps else 'n/a'
            self.get_logger().info(
                f'PLATOON STATS: mode {self.mode}, events {len(self.results)}, mean amplification {amp}, gap now '
                f'car 1 {gap1:.2f} m / robot {gap_r:.2f} m (target at 0.45 m/s {self.d0 + self.h * 0.45:.2f} m), '
                f'smallest gap car 1 {self.min_gap_car1:.2f} m, robot {self.min_gap_robot:.2f} m, '
                f'collisions {self.collisions}')


def main(args=None):
    rclpy.init(args=args)
    node = Platoon()
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
