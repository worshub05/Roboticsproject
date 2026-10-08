"""
ROS 2 Node for CNN-based steering inference
Subscribes to camera images, publishes steering commands
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
from pathlib import Path

from .lane_cnn_model import LaneCNN, preprocess_image


class CNNInferenceNode(Node):
    """ROS 2 Node for CNN steering inference"""

    def __init__(self):
        super().__init__('cnn_inference_node')

        # Declare parameters
        self.declare_parameter('model_path',
            str(Path.home() / 'ros2_ws' / 'models' / 'lane_cnn_retrained.pt'))
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('steering_topic', '/cmd_vel')
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('inference_rate', 30)

        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        self.steering_topic = self.get_parameter('steering_topic').value
        self.linear_speed = self.get_parameter('linear_speed').value
        inference_rate = self.get_parameter('inference_rate').value

        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        # Load model
        try:
            self.model = LaneCNN().to(self.device)
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                self.model.eval()
                param_count = self.model.count_parameters()
                self.get_logger().info(f"[MODEL] Loaded: {param_count} parameters")
            else:
                self.get_logger().warn(f"Model not found at {self.model_path}")
                self.get_logger().info("Continuing with untrained model")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = LaneCNN().to(self.device)

        # CV Bridge
        self.bridge = CvBridge()

        # Subscriptions
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # Publishers
        self.steering_publisher = self.create_publisher(
            Twist,
            self.steering_topic,
            10
        )

        # Inference rate control
        self.create_timer(1.0 / inference_rate, self.inference_loop)
        self.latest_image = None
        self.latest_steering = 0.0

        self.get_logger().info(
            f"CNN Inference Node started - "
            f"Camera: {camera_topic}, Output: {self.steering_topic}, "
            f"Rate: {inference_rate} Hz"
        )

    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def inference_loop(self):
        """Main inference loop"""
        if self.latest_image is None:
            return

        try:
            with torch.no_grad():
                # Preprocess
                image_tensor = preprocess_image(self.latest_image)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)

                # Inference
                steering_output = self.model(image_tensor)
                self.latest_steering = steering_output.cpu().item()

            # Publish steering command
            cmd_msg = Twist()
            cmd_msg.linear.x = self.linear_speed
            cmd_msg.angular.z = float(self.latest_steering)
            self.steering_publisher.publish(cmd_msg)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CNNInferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
