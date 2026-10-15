from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'lane_follower_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Diksha',
    maintainer_email='dikshajangam26@gmail.com',
    description='CNN-based lane following for warehouse robots',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cnn_inference_node = lane_follower_perception.cnn_inference_node:main',
        ],
    },
)
