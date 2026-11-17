#!/usr/bin/env python3
"""
Energy monitor: the EV energy model (ev_energy.py) running live on the robot's odometry.

Subscribes /odom. Publishes
  /ev/power   Float32  battery power now, W (+ = draws from the battery, - = charges it)
  /ev/energy  String   JSON with the totals so far (traction, recovered, friction brakes, distance)
and logs "ENERGY STATS: ..." every 5 s; run_stack.sh prints the last one when it stops.
Parameters: the EVParams fields (mass, c_rr, eta_drive, eta_regen, regen, p_aux, ...).
"""
import dataclasses
import json

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from .ev_energy import EnergyMeter, EVParams


class EnergyMonitor(Node):
    def __init__(self):
        super().__init__('energy_monitor')
        defaults = dataclasses.asdict(EVParams())
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.declare_parameter('log_period', 5.0)
        params = EVParams(**{k: type(v)(self.get_parameter(k).value) for k, v in defaults.items()})
        self.meter = EnergyMeter(params)
        self.power_pub = self.create_publisher(Float32, '/ev/power', 10)
        self.energy_pub = self.create_publisher(String, '/ev/energy', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 20)
        self.create_timer(float(self.get_parameter('log_period').value), self.log)
        self.get_logger().info(f'ENERGY: model ready: {params.mass} kg, C_rr {params.c_rr}, battery to wheel '
                               f'{params.eta_drive:.0%}, regen {"on, " + format(params.eta_regen, ".0%") if params.regen else "off"}'
                               f', computer {params.p_aux} W')

    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        p = self.meter.step(msg.twist.twist.linear.x, t)
        self.power_pub.publish(Float32(data=float(p)))

    def log(self):
        s = self.meter.summary()
        self.energy_pub.publish(String(data=json.dumps({k: round(v, 4) for k, v in s.items()})))
        if s['distance_m'] > 0.05:
            self.get_logger().info('ENERGY STATS: ' + self.meter.line())


def main(args=None):
    rclpy.init(args=args)
    node = EnergyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node.log()
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
