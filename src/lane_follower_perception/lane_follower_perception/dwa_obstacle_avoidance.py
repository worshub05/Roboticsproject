#!/usr/bin/env python3
"""
Dynamic Window Approach (DWA) obstacle avoidance.

Samples (v, w) pairs, simulates all candidate arcs at once with numpy,
rejects arcs that come too close to any LiDAR return, and publishes the
lowest-cost safe command on /cmd_vel.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class DWAObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('dwa_obstacle_avoidance')

        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_reverse_vel', 0.1)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('lin_vel_samples', 7)
        self.declare_parameter('ang_vel_samples', 31)
        self.declare_parameter('predict_time', 2.5)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('safety_radius', 0.8)
        self.declare_parameter('warning_radius', 1.0)
        self.declare_parameter('scan_range', 3.0)
        self.declare_parameter('weight_heading', 0.5)
        self.declare_parameter('weight_velocity', 1.0)
        self.declare_parameter('weight_obstacle', 2.0)
        self.declare_parameter('stop_penalty', 2.0)
        self.declare_parameter('reverse_penalty', 1.0)
        self.declare_parameter('weight_progress', 1.0)
        self.declare_parameter('collision_time', 1.2)
        self.declare_parameter('max_decel', 1.0)

        g = self.get_parameter
        self.max_v = g('max_linear_vel').value
        self.max_rev = g('max_reverse_vel').value
        self.max_w = g('max_angular_vel').value
        self.dt = g('dt').value
        self.steps = int(round(g('predict_time').value / self.dt))
        self.robot_radius = g('robot_radius').value
        self.safety = g('safety_radius').value
        self.warning = g('warning_radius').value
        self.scan_range = g('scan_range').value
        self.w_heading = g('weight_heading').value
        self.w_velocity = g('weight_velocity').value
        self.w_obstacle = g('weight_obstacle').value
        self.stop_penalty = g('stop_penalty').value
        self.reverse_penalty = g('reverse_penalty').value
        self.w_progress = g('weight_progress').value
        self.collision_steps = max(1, min(self.steps, int(round(g('collision_time').value / self.dt))))
        self.max_decel = g('max_decel').value

        # Candidate (v, w) grid, built once.
        v = np.linspace(-self.max_rev, self.max_v, int(g('lin_vel_samples').value))
        w = np.linspace(-self.max_w, self.max_w, int(g('ang_vel_samples').value))
        V, W = np.meshgrid(v, w, indexing='ij')
        self.cand_v = V.ravel()
        self.cand_w = W.ravel()

        # Precomputed arc times for the simulation.
        self.times = np.arange(1, self.steps + 1) * self.dt
        self.heading = self.cand_w[:, None] * self.times[None, :]

        self.desired = Twist()
        self.scan = None
        self.plan_count = 0
        self.plan_time = 0.0

        self.create_subscription(Twist, '/cmd_vel_desired', self.desired_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.active_pub = self.create_publisher(Bool, '/dwa/active', 10)

        self.get_logger().info(f'DWA initialised: {len(self.cand_v)} candidate arcs')

    def scan_callback(self, msg):
        self.scan = msg

    def desired_callback(self, msg):
        self.desired = msg
        self.plan()

    def obstacle_points(self):
        """LiDAR returns inside scan_range, in the robot frame, subsampled."""
        if self.scan is None:
            return np.zeros((0, 2))
        r = np.asarray(self.scan.ranges, dtype=float)
        angles = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment
        valid = np.isfinite(r) & (r > self.scan.range_min) & (r < self.scan_range)
        pts = np.stack([r[valid] * np.cos(angles[valid]),
                        r[valid] * np.sin(angles[valid])], axis=1)
        return pts[::2]

    def trajectories(self):
        """Positions of every candidate arc at each time step, shape (N, S, 2)."""
        v = self.cand_v[:, None]
        x = np.cumsum(v * np.cos(self.heading) * self.dt, axis=1)
        y = np.cumsum(v * np.sin(self.heading) * self.dt, axis=1)
        return np.stack([x, y], axis=2)

    def clearance(self, obs, P):
        """Closest approach of each arc: within the collision horizon, and over the full arc."""
        d = np.hypot(P[:, :, None, 0] - obs[None, None, :, 0],
                     P[:, :, None, 1] - obs[None, None, :, 1]).min(axis=2)  # (N, S)
        origin_clear = np.hypot(obs[:, 0], obs[:, 1]).min()
        near = np.minimum(d[:, :self.collision_steps].min(axis=1), origin_clear)
        full = np.minimum(d.min(axis=1), origin_clear)
        return near, full

    def plan(self):
        t0 = time.perf_counter()
        obs = self.obstacle_points()
        v_des = self.desired.linear.x
        w_des = self.desired.angular.z

        P = self.trajectories()
        # forward progress of each arc, normalised to roughly 0..1
        progress = P[:, -1, 0] / (self.max_v * self.steps * self.dt)
        if len(obs):
            clear, clear_full = self.clearance(obs, P)
        else:
            clear = np.full(len(self.cand_v), np.inf)
            clear_full = clear

        cost = (self.w_heading * np.abs(self.cand_w - w_des)
                + self.w_velocity * np.abs(self.cand_v - v_des)
                + self.w_obstacle * np.maximum(0.0, self.safety - clear_full) / self.safety
                + self.stop_penalty * (np.abs(self.cand_v) < 0.05)
                + self.reverse_penalty * (self.cand_v < 0.0)
                - self.w_progress * progress)
        cost[clear < self.robot_radius] = np.inf
        # only speeds the robot can still brake from before reaching the obstacle
        v_allowed = np.sqrt(2.0 * self.max_decel * np.maximum(0.0, clear - self.robot_radius))
        cost[np.abs(self.cand_v) > v_allowed] = np.inf

        i = int(np.argmin(cost))
        if np.isfinite(cost[i]):
            v, w = float(self.cand_v[i]), float(self.cand_w[i])
        else:
            # boxed in: back away slowly while turning toward the CNN's preferred side
            v = -self.max_rev
            w = 0.5 * self.max_w * (1.0 if w_des >= 0.0 else -1.0)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        nearest = float(np.min(np.hypot(obs[:, 0], obs[:, 1]))) if len(obs) else float('inf')
        self.active_pub.publish(Bool(data=nearest < self.warning))

        self.plan_time += time.perf_counter() - t0
        self.plan_count += 1
        if self.plan_count % 60 == 0:
            self.get_logger().info(
                f'DWA: v={v:+.2f} m/s w={w:+.2f} rad/s nearest={nearest:.2f} m '
                f'avg plan {1000.0 * self.plan_time / self.plan_count:.1f} ms')


def main(args=None):
    rclpy.init(args=args)
    node = DWAObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
