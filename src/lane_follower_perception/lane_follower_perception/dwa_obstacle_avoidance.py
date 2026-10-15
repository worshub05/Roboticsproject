#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool

class DWAObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('dwa_obstacle_avoidance')
        
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('safety_radius', 0.5)
        
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.safety_radius = self.get_parameter('safety_radius').value
        
        self.desired_vel_sub = self.create_subscription(
            Twist, '/cmd_vel_desired', self.desired_vel_callback, 10)
        self.threat_sub = self.create_subscription(
            Float32, '/obstacle/threat_level', self.threat_callback, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.active_pub = self.create_publisher(Bool, '/dwa/active', 10)
        
        self.current_threat = 0.0
        self.desired_vel = Twist()
        
        self.get_logger().info('🎯 DWA Obstacle Avoidance initialized')

    def desired_vel_callback(self, msg):
        self.desired_vel = msg
        self.publish_cmd_vel()

    def threat_callback(self, msg):
        self.current_threat = msg.data
        self.publish_cmd_vel()

    def publish_cmd_vel(self):
        cmd_vel = Twist()
        
        if self.current_threat > 0.1:
            # Reduce velocity based on threat level
            velocity_scale = 1.0 - self.current_threat
            cmd_vel.linear.x = self.desired_vel.linear.x * velocity_scale
            cmd_vel.angular.z = self.desired_vel.angular.z * velocity_scale
            active = True
        else:
            # Use desired velocity when safe
            cmd_vel = self.desired_vel
            active = False
        
        self.cmd_vel_pub.publish(cmd_vel)
        self.active_pub.publish(Bool(data=active))

def main():
    rclpy.init()
    node = DWAObstacleAvoidance()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
