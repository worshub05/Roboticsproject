#!/usr/bin/env python3
"""
Traffic lights with a V2I roadside unit, plus a ground-truth referee.

Infrastructure: fixed-time traffic lights at stop lines on our lane (parameter stop_s). Each light
is drawn in Gazebo (pole, red / amber / green lamp, white stop line) and its state is broadcast on
/v2i/spat (std_msgs/String, JSON modelled on the SAE J2735 SPaT + MAP messages) at 10 Hz.
The radio link can be made worse: loss (share of messages dropped), delay (s), blackout (s after
start when the unit stops sending; < 0 = never).

Referee (Gazebo ground truth, independent of the car): watches our robot and logs
  LIGHT: crossings of every stop line and the light's state at that moment (red = violation)
  LIGHT: every stop in front of a light, with the gap left to the line, and how long we waited
  V2I STATS: running totals
"""
import json
import math
import random

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from .track_map import to_frenet, track_path
from .v2i import LightSchedule, ahead, spat_message

ROBOT_FRONT = 0.15
LAMP = {'RED': ('1 0 0 1', 0.56), 'AMBER': ('1 0.6 0 1', 0.48), 'GREEN': ('0 1 0 1', 0.40)}


def pole_sdf(name):
    return f"""<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><static>true</static><link name="pole">
  <collision name="c"><pose>0 0 0.3 0 0 0</pose><geometry><cylinder><radius>0.02</radius><length>0.6</length></cylinder></geometry></collision>
  <visual name="v"><pose>0 0 0.3 0 0 0</pose><geometry><cylinder><radius>0.02</radius><length>0.6</length></cylinder></geometry>
    <material><ambient>0.2 0.2 0.2 1</ambient><diffuse>0.2 0.2 0.2 1</diffuse></material></visual>
  <visual name="housing"><pose>0 0 0.48 0 0 0</pose><geometry><box><size>0.06 0.07 0.26</size></box></geometry>
    <material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse></material></visual>
</link></model></sdf>"""


def lamp_sdf(name, colour):
    return f"""<?xml version="1.0"?>
<sdf version="1.6"><model name="{name}"><link name="lamp">
  <kinematic>true</kinematic><gravity>false</gravity>
  <visual name="v"><geometry><box><size>0.02 0.06 0.07</size></box></geometry>
    <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse><emissive>{colour}</emissive></material></visual>
</link></model></sdf>"""


def line_sdf(name):
    return """<?xml version="1.0"?>
<sdf version="1.6"><model name="%s"><static>true</static><link name="line">
  <visual name="v"><pose>0 0 0.006 0 0 0</pose><geometry><box><size>0.04 0.5 0.004</size></box></geometry>
    <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>0.8 0.8 0.8 1</emissive></material></visual>
</link></model></sdf>""" % name


class TrafficLights(Node):
    def __init__(self):
        super().__init__('traffic_light')
        for k, v in dict(stop_s=[12.0, 47.7], offsets=[0.0, 9.0], green=10.0, amber=3.0, red=12.0, rate=10.0,
                         loss=0.0, delay=0.0, blackout=-1.0, seed=1).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        offs = list(g('offsets')) + [0.0] * len(g('stop_s'))
        self.lights = [(i + 1, float(s), LightSchedule(g('green'), g('amber'), g('red'), float(offs[i])))
                       for i, s in enumerate(g('stop_s'))]
        self.loss, self.delay, self.blackout = float(g('loss')), float(g('delay')), float(g('blackout'))
        self.rnd = random.Random(int(g('seed')))
        self.t0, self.queue, self.sent, self.dropped = None, [], 0, 0
        self.shown = {}                          # light id -> lamp currently lit
        self.spawned, self.spawn_futures = False, []
        self.robot = None                        # s of the front bumper, d, speed
        self.prev_front = None
        self.stopped = {}                        # light id -> (time we stopped, gap)
        self.st = dict(green=0, amber=0, red=0, stops=0, gaps=[], waits=[])
        self.count = 0
        self.pub = self.create_publisher(String, '/v2i/spat', 10)
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(1.0 / float(g('rate')), self.tick)
        self.get_logger().info(
            'Traffic lights at s = ' + ', '.join(f'{s:.1f} m' for _, s, _ in self.lights) +
            f' (green {g("green")} s, amber {g("amber")} s, red {g("red")} s); V2I link: loss {self.loss:.0%}, '
            f'delay {self.delay} s' + (f', blackout after {self.blackout} s' if self.blackout >= 0 else ''))

    # ---------------------------------------------------------------- Gazebo drawing
    def spawn_one(self, name, xml, x, y, z, yaw):
        req = SpawnEntity.Request()
        req.name, req.xml, req.reference_frame = name, xml, 'world'
        req.initial_pose.position.x, req.initial_pose.position.y, req.initial_pose.position.z = float(x), float(y), float(z)
        req.initial_pose.orientation.z, req.initial_pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.spawn_futures.append(self.spawn_client.call_async(req))

    def lamp_pose(self, stop_s, state, lit):
        x, y, th = track_path(stop_s, -0.5)
        z = LAMP[state][1] if lit else -2.0
        # lamps on the side of the housing that faces the traffic coming towards the light
        return x - 0.035 * math.cos(th), y - 0.035 * math.sin(th), z, th

    def spawn(self):
        for lid, stop_s, _ in self.lights:
            x, y, th = track_path(stop_s, -0.5)
            self.spawn_one(f'light_{lid}_pole', pole_sdf(f'light_{lid}_pole'), x, y, 0.0, th)
            lx, ly, lth = track_path(stop_s, 0.0)
            self.spawn_one(f'light_{lid}_stop_line', line_sdf(f'light_{lid}_stop_line'), lx, ly, 0.0, lth)
            for state, (colour, _) in LAMP.items():
                px, py, pz, pth = self.lamp_pose(stop_s, state, False)
                self.spawn_one(f'light_{lid}_{state.lower()}', lamp_sdf(f'light_{lid}_{state.lower()}', colour),
                               px, py, pz, pth)
        self.spawned = True

    def move_lamp(self, lid, stop_s, state, lit):
        x, y, z, th = self.lamp_pose(stop_s, state, lit)
        req = SetEntityState.Request()
        req.state.name = f'light_{lid}_{state.lower()}'
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = float(x), float(y), float(z)
        req.state.pose.orientation.z, req.state.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        req.state.reference_frame = 'world'
        self.state_client.call_async(req)
    # ---------------------------------------------------------------- referee
    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        s, d = to_frenet(x, y)
        self.robot = (float(s[0]) + ROBOT_FRONT, float(d[0]), msg.twist.twist.linear.x)

    def referee(self, t):
        if self.robot is None:
            return
        front, d, v = self.robot
        for lid, stop_s, sched in self.lights:
            state, _, since = sched.state_at(t)
            gap = ahead(stop_s, front)
            if self.prev_front is not None and abs(d) < 0.3:
                before = ahead(stop_s, self.prev_front)
                if before > 0.0 >= gap and before - gap < 1.0:           # the front bumper crossed the line
                    self.st[state.lower()] += 1
                    if state == 'RED':
                        self.get_logger().info(f'LIGHT: ❌ RED LIGHT VIOLATION at light {lid} ({since:.1f} s into red, '
                                               f'{v:.2f} m/s)')
                    elif state == 'AMBER':
                        self.get_logger().info(f'LIGHT: 🟡 crossed light {lid} on amber ({since:.1f} s into amber, '
                                               f'{v:.2f} m/s)')
                    else:
                        self.get_logger().info(f'LIGHT: ✅ crossed light {lid} on green at {v:.2f} m/s')
            if 0.0 < gap < 1.0 and abs(v) < 0.02 and state != 'GREEN' and lid not in self.stopped:
                self.stopped[lid] = (t, gap)
                self.st['stops'] += 1
                self.st['gaps'].append(gap)
                self.get_logger().info(f'LIGHT: 🛑 stopped {gap * 100:.1f} cm before light {lid} ({state})')
            if lid in self.stopped and abs(v) > 0.05:
                t_stop, _ = self.stopped.pop(lid)
                self.st['waits'].append(t - t_stop)
                self.get_logger().info(f'LIGHT: light {lid} {state.lower()}: drove off after waiting {t - t_stop:.1f} s')
        self.prev_front = front

    # ---------------------------------------------------------------- main loop
    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        # broadcast (with the simulated radio link)
        if self.blackout < 0 or t < self.blackout:
            self.sent += 1
            if self.rnd.random() >= self.loss:
                self.queue.append((now + self.delay, spat_message(self.lights, t)))
            else:
                self.dropped += 1
        while self.queue and self.queue[0][0] <= now:
            self.pub.publish(String(data=self.queue.pop(0)[1]))
        # draw the lights
        if not self.spawned:
            if self.spawn_client.service_is_ready():
                self.spawn()
        elif self.state_client.service_is_ready() and all(f.done() for f in self.spawn_futures):
            for lid, stop_s, sched in self.lights:
                state = sched.state_at(t)[0]
                if self.shown.get(lid) != state:
                    for st in LAMP:
                        self.move_lamp(lid, stop_s, st, st == state)
                    self.shown[lid] = state
        self.referee(t)
        self.count += 1
        if self.count % 50 == 0:
            s = self.st
            gaps = f'{sum(s["gaps"]) / len(s["gaps"]) * 100:.1f} cm (range {min(s["gaps"]) * 100:.1f}-' \
                   f'{max(s["gaps"]) * 100:.1f})' if s['gaps'] else 'n/a'
            waits = f'{sum(s["waits"]) / len(s["waits"]):.1f} s' if s['waits'] else 'n/a'
            self.get_logger().info(
                f'V2I STATS: crossings green {s["green"]}, amber {s["amber"]}, RED {s["red"]}; stops {s["stops"]}, '
                f'gap to the line {gaps}, mean wait {waits}; messages sent {self.sent}, lost {self.dropped}')


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLights()
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
