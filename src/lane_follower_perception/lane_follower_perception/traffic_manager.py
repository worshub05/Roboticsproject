#!/usr/bin/env python3
"""
Traffic manager: moves the other cars in the road world.

Scenarios (parameter 'scenario'):
  oncoming  - two cars drive towards us in the other lane, looping
  follow    - a slower car ahead in our lane (0.25 m/s), which later speeds up and leaves
  crossing  - a car crosses the road at the junction (x = 25) as we approach it
  all       - all of the above in one run

Cars are moved with Gazebo's /gazebo/set_entity_state service.
"""
import math

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from nav_msgs.msg import Odometry

PARK = {'lead_car': (-50.0, 5.0), 'oncoming_car_1': (-50.0, 7.0),
        'oncoming_car_2': (-50.0, 9.0), 'crossing_car': (-50.0, 11.0)}


class TrafficManager(Node):
    def __init__(self):
        super().__init__('traffic_manager')

        self.declare_parameter('scenario', 'all')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('lead_start_x', 5.0)
        self.declare_parameter('lead_slow_speed', 0.25)
        self.declare_parameter('lead_fast_speed', 0.7)
        self.declare_parameter('lead_slow_time', 45.0)
        self.declare_parameter('oncoming_speed', 0.4)
        self.declare_parameter('crossing_x', 25.0)
        self.declare_parameter('crossing_trigger_x', 21.0)
        self.declare_parameter('crossing_speed', 0.4)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.scenario = g('scenario')
        self.p = {n: g(n) for n in ('lead_start_x', 'lead_slow_speed', 'lead_fast_speed', 'lead_slow_time',
                                    'oncoming_speed', 'crossing_x', 'crossing_trigger_x', 'crossing_speed')}

        use = lambda s: self.scenario in ('all', s)  # noqa: E731
        # name -> [x, y, yaw, active]
        self.cars = {}
        if use('follow'):
            self.cars['lead_car'] = [self.p['lead_start_x'], 0.0, 0.0, True]
        if use('oncoming'):
            self.cars['oncoming_car_1'] = [30.0, 0.6, math.pi, True]
            self.cars['oncoming_car_2'] = [50.0, 0.6, math.pi, True]
        if use('crossing'):
            self.cars['crossing_car'] = [self.p['crossing_x'], -3.0, math.pi / 2, False]
        self.unused = [n for n in PARK if n not in self.cars]

        self.robot_x = 0.0
        self.t0 = None
        self.prev_t = None
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(1.0 / g('rate'), self.tick)
        self.get_logger().info(f'Traffic manager: scenario "{self.scenario}", cars {list(self.cars)}')

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x

    def send(self, name, x, y, yaw):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.reference_frame = 'world'
        self.client.call_async(req)

    def tick(self):
        if not self.client.service_is_ready():
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = self.prev_t = now
            for name in self.unused:  # park cars this scenario does not use
                self.send(name, *PARK[name], 0.0)
        t, dt = now - self.t0, now - self.prev_t
        self.prev_t = now
        if dt <= 0.0:
            return

        for name, car in self.cars.items():
            x, y, yaw, active = car
            if name == 'lead_car':
                if not active:
                    continue  # already left the road
                speed = self.p['lead_slow_speed'] if t < self.p['lead_slow_time'] else self.p['lead_fast_speed']
                x += speed * dt
                if x > 66.0:
                    (x, y), active = PARK[name], False
            elif name.startswith('oncoming'):
                x -= self.p['oncoming_speed'] * dt
                if x < -4.0:
                    x += 60.0
            elif name == 'crossing_car':
                if not active and self.robot_x > self.p['crossing_trigger_x']:
                    active = True
                    self.get_logger().info('Traffic: crossing car starts')
                if active and y < 4.0:
                    y += self.p['crossing_speed'] * dt
            car[:] = [x, y, yaw, active]
            self.send(name, x, y, yaw)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
