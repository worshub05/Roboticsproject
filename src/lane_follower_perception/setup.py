from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'lane_follower_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.py'))),
        ('share/' + package_name + '/robots', glob(os.path.join('robots', '*.urdf'))),
        ('share/' + package_name + '/worlds', glob(os.path.join('worlds', '*.world'))),
        ('share/' + package_name + '/models', glob(os.path.join('models', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shub_2605',
    maintainer_email='shub_2605@example.com',
    description='CNN lane following perception node for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cnn_inference_node.py = lane_follower_perception.cnn_inference_node:main',
            'lidar_obstacle_detector.py = lane_follower_perception.lidar_obstacle_detector:main',
            'dwa_obstacle_avoidance.py = lane_follower_perception.dwa_obstacle_avoidance:main',
        ],
    },
)
