#!/usr/bin/env python3
"""
Lead car on the oval track, for testing adaptive cruise control on bends.

Spawns a small blue car in our lane and drives it along the lane centre with Gazebo's
/gazebo/set_entity_state. Also measures the TRUE bumper-to-bumper gap along the lane
(from Gazebo positions, not from the LiDAR), so the ACC can be checked against ground truth.

Parameters:
  lead_start_s   where the lead car starts, metres along our lane from the robot's start
                 (24 = a few metres into the first bend)
  lead_speed     m/s (0 = a stopped car)
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from .collect_track_data import L, R, track_path

PER = 2 * L + 2 * math.pi * R
CAR_HALF_LENGTH = 0.25
ROBOT_FRONT = 0.15
CAR_SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="track_lead_car"><link name="body">
  <kinematic>true</kinematic><gravity>false</gravity>
  <inertial><mass>5.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
    <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
  <collision name="collision"><pose>0 0 0.225 0 0 0</pose>
    <geometry><box><size>0.5 0.25 0.35</size></box></geometry></collision>
  <visual name="body"><pose>0 0 0.18 0 0 0</pose><geometry><box><size>0.5 0.25 0.26</size></box></geometry>
    <material><ambient>0.1 0.3 0.8 1</ambient><diffuse>0.1 0.3 0.8 1</diffuse></material></visual>
  <visual name="cabin"><pose>-0.03 0 0.36 0 0 0</pose><geometry><box><size>0.28 0.22 0.1</size></box></geometry>
    <material><ambient>0.15 0.15 0.2 1</ambient><diffuse>0.15 0.15 0.2 1</diffuse></material></visual>
</link></model></sdf>"""


class TrackTraffic(Node):
    def __init__(self):
        super().__init__('track_traffic')
        self.declare_parameter('lead_start_s', 24.0)
        self.declare_parameter('lead_speed', 0.15)
        self.declare_parameter('rate', 20.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.s = float(g('lead_start_s'))
        self.v = float(g('lead_speed'))

        ss = np.arange(0.0, PER, 0.02)
        self.ss = ss
        self.path = np.array([track_path(si, 0.0) for si in ss])
        self.robot = None            # (s along lane, speed)
        self.spawned = False
        self.t_prev = None
        self.min_gap = math.inf
        self.collided = False
        self.count = 0

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.gap_pub = self.create_publisher(Float32, '/traffic/true_gap', 10)
        self.create_timer(1.0 / g('rate'), self.tick)
        self.get_logger().info(f'Track traffic: lead car at s={self.s:.1f} m, {self.v:.2f} m/s')

    def odom_callback(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        i = int(np.argmin((self.path[:, 0] - x) ** 2 + (self.path[:, 1] - y) ** 2))
        self.robot = (float(self.ss[i]), msg.twist.twist.linear.x)

    def spawn(self):
        x, y, th = track_path(self.s, 0.0)
        req = SpawnEntity.Request()
        req.name = 'track_lead_car'
        req.xml = CAR_SDF
        req.initial_pose.position.x, req.initial_pose.position.y = float(x), float(y)
        req.initial_pose.orientation.z, req.initial_pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        req.reference_frame = 'world'
        self.spawn_client.call_async(req)
        self.spawned = True
        self.get_logger().info('Track traffic: lead car spawned')

    def move(self):
        x, y, th = track_path(self.s, 0.0)
        req = SetEntityState.Request()
        req.state.name = 'track_lead_car'
        req.state.pose.position.x, req.state.pose.position.y = float(x), float(y)
        req.state.pose.orientation.z, req.state.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
        req.state.reference_frame = 'world'
        self.state_client.call_async(req)

    def tick(self):
        if not self.spawned:
            if self.spawn_client.service_is_ready():
                self.spawn()
            return
        if not self.state_client.service_is_ready():
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.t_prev is None else now - self.t_prev
        self.t_prev = now
        self.s = (self.s + self.v * dt) % PER
        self.move()

        if self.robot is None:
            return
        s_robot, v_robot = self.robot
        gap = (self.s - s_robot) % PER - CAR_HALF_LENGTH - ROBOT_FRONT
        self.gap_pub.publish(Float32(data=float(gap)))
        self.min_gap = min(self.min_gap, gap)
        if gap < 0.02 and not self.collided:
            self.collided = True
            self.get_logger().info('TRAFFIC: ❌ COLLISION with the lead car')
        self.count += 1
        if self.count % 40 == 0:
            where = 'bend' if (L <= s_robot < L + math.pi * R or s_robot >= 2 * L + math.pi * R) else 'straight'
            self.get_logger().info(f'TRAFFIC: true gap {gap:.2f} m, robot {v_robot:.2f} m/s on the {where}, '
                                   f'lead {self.v:.2f} m/s, smallest gap so far {self.min_gap:.2f} m')


def main(args=None):
    rclpy.init(args=args)
    node = TrackTraffic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        print(f'TRAFFIC SUMMARY: smallest true gap {node.min_gap:.2f} m, '
              f'collision: {"YES" if node.collided else "no"}', flush=True)
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
