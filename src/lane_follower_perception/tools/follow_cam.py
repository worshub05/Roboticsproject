#!/usr/bin/env python3
"""
Chase camera for recording runs: spawns an invisible camera in Gazebo, keeps it behind and above the
robot (moved with /gazebo/set_entity_state from /odom) and writes what it sees to an mp4 file.
It has no visual and no collision, so the robot's own camera and LiDAR do not see it.

  python3 tools/follow_cam.py --ros-args -p use_sim_time:=true -p out:=/tmp/run.mp4 -p title:="CACC platoon"
SIGTERM or Ctrl+C closes the video file.
"""
import math
import signal

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

SDF = """<?xml version="1.0"?>
<sdf version="1.6"><model name="follow_cam"><static>false</static>
<link name="link"><gravity>false</gravity><kinematic>true</kinematic>
<inertial><mass>0.1</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
<sensor name="cam" type="camera"><always_on>true</always_on><update_rate>{fps}</update_rate>
<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
<clip><near>0.05</near><far>80</far></clip></camera>
<plugin name="follow_cam_plugin" filename="libgazebo_ros_camera.so"><ros><namespace>/follow_cam</namespace></ros>
<camera_name>cam</camera_name><frame_name>follow_cam</frame_name></plugin>
</sensor></link></model></sdf>"""


class FollowCam(Node):
    def __init__(self):
        super().__init__('follow_cam')
        for k, v in dict(out='/tmp/follow_cam.mp4', title='', width=960, height=540, fps=15, fov=1.25,
                         back=1.5, up=1.0, ahead=0.6).items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out, self.title = str(g('out')), str(g('title'))
        self.w, self.h, self.fps, self.fov = int(g('width')), int(g('height')), int(g('fps')), float(g('fov'))
        self.back, self.up, self.ahead = float(g('back')), float(g('up')), float(g('ahead'))
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.spawn_future, self.sub, self.writer = None, None, None
        self.yaw, self.v, self.t0, self.frames = None, 0.0, None, 0
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(0.5, self.setup)

    def setup(self):
        if self.spawn_future is None:
            if self.spawn_client.service_is_ready():
                req = SpawnEntity.Request()
                req.name = 'follow_cam'
                req.xml = SDF.format(fps=self.fps, fov=self.fov, w=self.w, h=self.h)
                req.initial_pose.position.z = 3.0
                self.spawn_future = self.spawn_client.call_async(req)
            return
        if self.sub is None and self.spawn_future.done():
            for name, types in self.get_topic_names_and_types():
                if 'follow_cam' in name and name.endswith('image_raw') and 'sensor_msgs/msg/Image' in types:
                    self.sub = self.create_subscription(Image, name, self.image_callback, qos_profile_sensor_data)
                    self.get_logger().info(f'FOLLOW CAM: recording {name} to {self.out}')

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.v = msg.twist.twist.linear.x
        if self.yaw is None:
            self.yaw = yaw
        self.yaw += 0.08 * math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))   # smooth turns
        if self.spawn_future is None or not self.spawn_future.done() or not self.state_client.service_is_ready():
            return
        pitch = math.atan2(self.up - 0.1, self.back + self.ahead)        # look at a point ahead of the robot
        req = SetEntityState.Request()
        req.state.name = 'follow_cam'
        req.state.reference_frame = 'world'
        req.state.pose.position.x = p.x - self.back * math.cos(self.yaw)
        req.state.pose.position.y = p.y - self.back * math.sin(self.yaw)
        req.state.pose.position.z = self.up
        cy, sy, cp, sp = math.cos(self.yaw / 2), math.sin(self.yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
        o = req.state.pose.orientation                                   # roll 0, pitch down, yaw
        o.w, o.x, o.y, o.z = cy * cp, -sy * sp, cy * sp, sy * cp
        self.state_client.call_async(req)

    def image_callback(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (msg.width, msg.height))
        for text, y, scale in ((self.title, 34, 0.8), (f't = {t - self.t0:5.1f} s   speed {self.v:.2f} m/s', 64, 0.6)):
            if text:
                cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        # the camera may deliver fewer frames than fps (slow rendering): repeat each frame for the sim
        # time it covers, so the video plays at the real speed of the simulation
        n = 1 if self.frames == 0 else max(0, round((t - self.t0) * self.fps) - self.frames)
        for _ in range(min(n, 3 * self.fps)):
            self.writer.write(img)
            self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(f'FOLLOW CAM: {self.frames} frames written to {self.out}', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = FollowCam()

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    node.close()
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
