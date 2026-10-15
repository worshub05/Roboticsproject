#!/usr/bin/env python3
"""
Differential Drive Controller
Converts CNN steering angle to left/right wheel velocities
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DifferentialDriveController(Node):
    """
    Converts steering predictions to differential wheel commands
    CNN outputs steering angle → controller converts to wheel velocities
    """

    def __init__(self):
        super().__init__('differential_drive_controller')

        # Parameters
        self.linear_speed = 0.5  # Forward speed m/s
        self.wheel_separation = 0.3  # Distance between wheels (m)

        # Subscriptions
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/robot/odom',
            self.odom_callback,
            10
        )

        # Publishers
        self.drive_publisher = self.create_publisher(
            Twist,
            '/robot/cmd_vel',
            10
        )

        # State
        self.current_steering = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        self.get_logger().info("Differential Drive Controller started")
        self.get_logger().info(f"  Wheel separation: {self.wheel_separation}m")
        self.get_logger().info(f"  Linear speed: {self.linear_speed} m/s")

    def cmd_vel_callback(self, msg):
        """
        Receive steering angle from CNN
        Convert to differential drive commands
        """
        # Extract steering angle (CNN outputs to angular.z)
        self.current_steering = msg.angular.z

        # Kinematic conversion
        # For differential drive:
        # v_left = v_linear - (omega * L/2)
        # v_right = v_linear + (omega * L/2)
        # where omega = v_linear * tan(steering_angle) / wheelbase

        # Simplified: use steering angle to modulate angular velocity
        linear = self.linear_speed

        # Scale steering angle to angular velocity
        # steering angle range: [-1.2, 1.2] → angular velocity range
        angular = self.current_steering * 0.5  # Smooth scaling factor

        # Create differential drive command
        drive_cmd = Twist()
        drive_cmd.linear.x = linear
        drive_cmd.angular.z = angular

        # Publish command to differential drive plugin
        self.drive_publisher.publish(drive_cmd)

    def odom_callback(self, msg):
        """Track robot position from odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Simple yaw extraction
        self.robot_theta = 2 * (qw * qz + qx * qy)


def main(args=None):
    rclpy.init(args=args)
    controller = DifferentialDriveController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
