#!/usr/bin/env python3
"""
Test camera publisher - sends synthetic lane images for testing line follower
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('test_camera_publisher')
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.cv_bridge = CvBridge()
        
        # Create synthetic lane image
        self.frame_count = 0
        
        # Timer to publish at ~30 Hz
        self.timer = self.create_timer(0.033, self.publish_image)
        self.get_logger().info('Test camera publisher started')

    def publish_image(self):
        """Generate and publish a synthetic lane image"""
        # Create blank image (RGB)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a white lane in the middle
        lane_left = 280
        lane_right = 360
        
        # Draw lane markings
        cv2.line(img, (lane_left, 240), (lane_left, 480), (255, 255, 255), 3)
        cv2.line(img, (lane_right, 240), (lane_right, 480), (255, 255, 255), 3)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Vary the lane position slightly to simulate movement
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            lane_offset = int(30 * np.sin(self.frame_count / 100.0))
            cv2.line(img, (lane_left + lane_offset, 240), (lane_left + lane_offset, 480), (255, 255, 255), 3)
            cv2.line(img, (lane_right + lane_offset, 240), (lane_right + lane_offset, 480), (255, 255, 255), 3)
        
        # Convert to ROS message and publish
        ros_image = self.cv_bridge.cv2_to_imgmsg(img, encoding='rgb8')
        self.publisher.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    publisher = CameraPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
