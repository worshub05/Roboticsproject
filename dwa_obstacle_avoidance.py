#!/usr/bin/env python3
"""
Dynamic Window Approach (DWA) Obstacle Avoidance Controller
- Integrates with CNN lane-following steering
- Modifies velocity commands to avoid obstacles
- Implements trajectory sampling and cost evaluation
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
import math

class DWAObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('dwa_obstacle_avoidance')
        
        # DWA Parameters
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('lin_vel_resolution', 0.1)
        self.declare_parameter('ang_vel_resolution', 0.2)
        self.declare_parameter('predict_time', 1.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('safety_radius', 0.5)
        
        # Cost weights
        self.declare_parameter('weight_obstacle', 1.0)
        self.declare_parameter('weight_velocity', 0.1)
        self.declare_parameter('weight_heading', 0.1)
        
        # Get parameters
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.lin_vel_res = self.get_parameter('lin_vel_resolution').value
        self.ang_vel_res = self.get_parameter('ang_vel_resolution').value
        self.predict_time = self.get_parameter('predict_time').value
        self.dt = self.get_parameter('dt').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.safety_radius = self.get_parameter('safety_radius').value
        
        self.weight_obs = self.get_parameter('weight_obstacle').value
        self.weight_vel = self.get_parameter('weight_velocity').value
        self.weight_heading = self.get_parameter('weight_heading').value
        
        # Subscriptions
        self.desired_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel_desired',
            self.desired_velocity_callback,
            10
        )
        self.min_distance_sub = self.create_subscription(
            Float32,
            '/obstacle/min_distance',
            self.min_distance_callback,
            10
        )
        self.threat_level_sub = self.create_subscription(
            Float32,
            '/obstacle/threat_level',
            self.threat_level_callback,
            10
        )
        self.obstacle_detected_sub = self.create_subscription(
            Bool,
            '/obstacle/detected',
            self.obstacle_detected_callback,
            10
        )
        
        # Publisher for safe velocity command
        self.safe_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Publisher for avoidance status
        self.avoidance_active_pub = self.create_publisher(
            Bool,
            '/dwa/active',
            10
        )
        
        # State
        self.desired_vel = Twist()
        self.current_angular_z = 0.0
        self.min_distance = float('inf')
        self.threat_level = 0.0
        self.obstacle_detected = False
        self.avoidance_active = False
        
        # Control loop
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('DWA Obstacle Avoidance initialized')

    def desired_velocity_callback(self, msg: Twist):
        """Receive desired velocity from CNN lane follower"""
        self.desired_vel = msg

    def min_distance_callback(self, msg: Float32):
        """Update minimum obstacle distance"""
        self.min_distance = msg.data

    def threat_level_callback(self, msg: Float32):
        """Update threat level"""
        self.threat_level = msg.data

    def obstacle_detected_callback(self, msg: Bool):
        """Update obstacle detection status"""
        self.obstacle_detected = msg.data

    def control_loop(self):
        """Main DWA control loop"""
        if not self.obstacle_detected:
            self.safe_vel_pub.publish(self.desired_vel)
            self.avoidance_active_pub.publish(Bool(data=False))
            return
        
        self.avoidance_active = True
        
        v_min = 0.0
        v_max = self.max_linear_vel
        w_min = -self.max_angular_vel
        w_max = self.max_angular_vel
        
        v_samples = np.arange(v_min, v_max, self.lin_vel_res)
        w_samples = np.arange(w_min, w_max, self.ang_vel_res)
        
        best_cost = float('inf')
        best_v = 0.0
        best_w = 0.0
        
        for v in v_samples:
            for w in w_samples:
                collision = self.check_collision(v, w)
                
                if collision:
                    cost = float('inf')
                else:
                    obs_cost = self.calculate_obstacle_cost(v, w)
                    vel_cost = self.calculate_velocity_cost(v)
                    heading_cost = self.calculate_heading_cost(w)
                    
                    cost = (self.weight_obs * obs_cost + 
                           self.weight_vel * vel_cost + 
                           self.weight_heading * heading_cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
        
        safe_vel = Twist()
        safe_vel.linear.x = best_v
        safe_vel.angular.z = best_w
        
        self.get_logger().debug(
            f'DWA: v={best_v:.2f}, w={best_w:.2f}, '
            f'threat={self.threat_level:.2f}, '
            f'min_dist={self.min_distance:.2f}'
        )
        
        self.safe_vel_pub.publish(safe_vel)
        
        avoidance_msg = Bool()
        avoidance_msg.data = self.avoidance_active
        self.avoidance_active_pub.publish(avoidance_msg)

    def check_collision(self, v: float, w: float) -> bool:
        """Check if trajectory collides with obstacles"""
        x, y = 0.0, 0.0
        theta = 0.0
        
        steps = int(self.predict_time / self.dt)
        for _ in range(steps):
            if abs(w) < 1e-6:
                x += v * self.dt * math.cos(theta)
                y += v * self.dt * math.sin(theta)
            else:
                radius = v / w
                theta += w * self.dt
                x += radius * (math.sin(theta) - math.sin(theta - w * self.dt))
                y += radius * (-math.cos(theta) + math.cos(theta - w * self.dt))
            
            distance_to_robot = math.sqrt(x**2 + y**2)
            if distance_to_robot - self.robot_radius < self.safety_radius:
                return True
        
        return False

    def calculate_obstacle_cost(self, v: float, w: float) -> float:
        """Cost function for obstacle avoidance"""
        if self.min_distance < self.robot_radius:
            return float('inf')
        
        if self.min_distance < self.safety_radius:
            return (self.safety_radius - self.min_distance) / self.safety_radius
        
        return 0.0

    def calculate_velocity_cost(self, v: float) -> float:
        """Cost function for velocity smoothness"""
        desired_v = self.desired_vel.linear.x
        return abs(v - desired_v) / (self.max_linear_vel + 1e-6)

    def calculate_heading_cost(self, w: float) -> float:
        """Cost function for heading alignment"""
        return abs(w) / (self.max_angular_vel + 1e-6)


def main(args=None):
    rclpy.init(args=args)
    controller = DWAObstacleAvoidance()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Node interrupted')
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
