#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for line follower node."""
    
    # Get package directory
    pkg_dir = get_package_share_directory('line_follower_fsdetect')
    
    # Path to stop sign image in package share directory
    stop_sign_path = os.path.join(pkg_dir, '..', 'line_follower_fsdetect', 'stop_sign.png')
    
    # Line follower node
    line_follower_node = Node(
        package='line_follower_fsdetect',
        executable='line_follower',
        name='line_follower_node',
        output='screen',
        parameters=[
            {'camera_topic': '/camera/image_raw'},
            {'cmd_vel_topic': '/cmd_vel'},
            {'canny_threshold1': 50},
            {'canny_threshold2': 150},
            {'max_linear_vel': 0.15},
            {'max_angular_vel': 1.0},
            {'stop_sign_path': ''},
            {'show_debug': False},
        ]
    )
    
    return LaunchDescription([
        line_follower_node,
    ])
