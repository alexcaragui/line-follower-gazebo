#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

from .freespace import get_freespace_angle


class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('canny_threshold1', 50)
        self.declare_parameter('canny_threshold2', 150)
        self.declare_parameter('max_linear_vel', 0.2)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('stop_sign_path', '')
        self.declare_parameter('show_debug', True)
        
        # Get parameters
        camera_topic = self.get_parameter('camera_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        stop_sign_path = self.get_parameter('stop_sign_path').value
        self.show_debug = self.get_parameter('show_debug').value
        
        # Initialize stop sign detection
        self.stop_sign_detected = False
        self.stop_sign_template = None
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()
        self.stop_sign_kp = None
        self.stop_sign_des = None
        
        # Load stop sign template if provided
        if stop_sign_path and os.path.exists(stop_sign_path):
            self.stop_sign_template = cv2.imread(stop_sign_path)
            if self.stop_sign_template is not None:
                gray_template = cv2.cvtColor(self.stop_sign_template, cv2.COLOR_BGR2GRAY)
                self.stop_sign_kp, self.stop_sign_des = self.sift.detectAndCompute(gray_template, None)
                self.get_logger().info(f'Loaded stop sign template from {stop_sign_path}')
                self.get_logger().info(f'Stop sign has {len(self.stop_sign_kp)} SIFT keypoints')
            else:
                self.get_logger().warn(f'Failed to load stop sign from {stop_sign_path}')
        
        # Create subscriptions and publishers
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )
        
        self.cv_bridge = CvBridge()
        self.get_logger().info(f'Subscribed to {camera_topic}')
        self.get_logger().info(f'Publishing to {cmd_vel_topic}')
        self.get_logger().info('Line Follower node started. Close debug windows to continue.')
    
    def detect_stop_sign_sift(self, frame_gray):
        """Detect stop sign using SIFT matching. Returns True if detected, False otherwise."""
        if self.stop_sign_des is None or len(self.stop_sign_kp) < 4:
            return False
        
        try:
            # Detect SIFT keypoints in current frame
            kp, des = self.sift.detectAndCompute(frame_gray, None)
            
            if des is None or len(kp) < 4:
                return False
            
            # Match keypoints
            matches = self.bf_matcher.knnMatch(self.stop_sign_des, des, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # If enough good matches found, stop sign detected
            if len(good_matches) >= 5:
                return True
            
            return False
        except Exception as e:
            self.get_logger().warn(f'Error in SIFT detection: {e}')
            return False
    
    def image_callback(self, msg):
        """Process camera image and calculate steering command."""
        try:
            # Convert ROS image to OpenCV format
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Process image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for stop sign using SIFT
        stop_sign_detected = self.detect_stop_sign_sift(gray)
        
        # Create debug image (BGR copy of original)
        debug_img = frame.copy()
        
        # If stop sign detected, stop the robot
        if stop_sign_detected:
            self.get_logger().info('STOP SIGN DETECTED! Stopping robot.')
            self.stop_sign_detected = True
            
            # Publish zero velocity
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            
            # Display stop message
            cv2.putText(
                debug_img,
                'STOP SIGN DETECTED!',
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
            cv2.rectangle(debug_img, (20, 20), (debug_img.shape[1]-20, debug_img.shape[0]-20), (0, 0, 255), 3)
        else:
            self.stop_sign_detected = False
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                gray,
                self.canny_threshold1,
                self.canny_threshold2
            )
            
            # Get steering angle from free space detection
            try:
                angle_deg, centroid = get_freespace_angle(edges, debug_img)
            except Exception as e:
                self.get_logger().warn(f'Error in free space detection: {e}')
                return
            
            # Normalize angle to [-90, 90] degrees
            steering_angle = np.clip(angle_deg, -90, 90)
            
            # Convert angle to angular velocity (rad/s)
            angular_vel = (steering_angle / 90.0) * self.max_angular_vel
            
            # Simple strategy: always move forward, adjust rotation
            linear_vel = self.max_linear_vel
            
            # Publish velocity command
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -angular_vel
            self.cmd_vel_pub.publish(twist)
            
            # Show debug info on image
            cv2.putText(
                debug_img,
                f'Angle: {steering_angle:.1f} deg',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                debug_img,
                f'Linear: {linear_vel:.2f} m/s, Angular: {angular_vel:.2f} rad/s',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Canny Edges', edges)
        
        # Display images only if debug is enabled
        if self.show_debug:
            cv2.imshow('Original', frame)
            cv2.imshow('Debug - Free Space', debug_img)
            
            # Exit on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.get_logger().info('ESC pressed. Shutting down...')
                raise KeyboardInterrupt()
        else:
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    except Exception as e:
        node.get_logger().error(f'Error: {e}')
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
