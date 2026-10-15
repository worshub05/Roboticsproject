#!/usr/bin/env python3
"""
CNN Inference Node for ROS 2
Subscribes to camera, publishes steering commands
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch

from .lane_cnn_model import load_model, infer_steering


class CNNInferenceNode(Node):
    """ROS 2 node for CNN-based steering inference"""
    
    def __init__(self):
        super().__init__('cnn_inference_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '/home/shub_2605/ros2_ws/models/lane_cnn.pt')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('steering_topic', '/cmd_vel')
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('inference_rate', 30)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        camera_topic = self.get_parameter('camera_topic').value
        steering_topic = self.get_parameter('steering_topic').value
        self.linear_speed = self.get_parameter('linear_speed').value
        inference_rate = self.get_parameter('inference_rate').value
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'🔧 Using device: {self.device}')
        
        # Load CNN model
        self.get_logger().info(f'📦 Loading model from: {model_path}')
        self.model = load_model(model_path, device=self.device)
        self.get_logger().info(f'✅ Model loaded successfully')
        
        # Initialize image bridge
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.get_logger().info(f'📷 Subscribing to: {camera_topic}')
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self._image_callback,
            10  # QoS depth
        )
        
        # Publish steering commands
        self.get_logger().info(f'🎮 Publishing to: {steering_topic}')
        self.publisher = self.create_publisher(
            Twist,
            steering_topic,
            10
        )
        
        # Statistics
        self.frame_count = 0
        self.get_logger().info(f'✅ CNN Inference Node initialized')
        self.get_logger().info(f'   Speed: {self.linear_speed} m/s')
        self.get_logger().info(f'   Rate: {inference_rate} Hz')
    
    def _image_callback(self, msg):
        """Camera image callback"""
        try:
            # Convert ROS message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Run inference
            steering = infer_steering(self.model, cv_image, device=self.device)
            
            # Create Twist message
            cmd = Twist()
            cmd.linear.x = self.linear_speed
            cmd.angular.z = float(steering)
            
            # Publish
            self.publisher.publish(cmd)
            
            # Log periodically
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'📊 Frames: {self.frame_count} | '
                    f'Steering: {steering:+.3f} rad | '
                    f'Speed: {cmd.linear.x} m/s'
                )
        
        except Exception as e:
            self.get_logger().error(f'❌ Inference error: {str(e)}')
            self.get_logger().error(f'   Image shape: {msg.width}x{msg.height}')


def main():
    rclpy.init()
    node = CNNInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
