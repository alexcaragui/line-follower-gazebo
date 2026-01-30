#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time

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
        # Traffic light HSV ranges (defaults similar to provided yaml)
        self.declare_parameter('red.hue_l', 0)
        self.declare_parameter('red.hue_h', 24)
        self.declare_parameter('red.saturation_l', 200)
        self.declare_parameter('red.saturation_h', 255)
        self.declare_parameter('red.lightness_l', 200)
        self.declare_parameter('red.lightness_h', 255)

        self.declare_parameter('yellow.hue_l', 19)
        self.declare_parameter('yellow.hue_h', 33)
        self.declare_parameter('yellow.saturation_l', 237)
        self.declare_parameter('yellow.saturation_h', 255)
        self.declare_parameter('yellow.lightness_l', 231)
        self.declare_parameter('yellow.lightness_h', 255)

        self.declare_parameter('green.hue_l', 40)
        self.declare_parameter('green.hue_h', 113)
        self.declare_parameter('green.saturation_l', 210)
        self.declare_parameter('green.saturation_h', 255)
        self.declare_parameter('green.lightness_l', 131)
        self.declare_parameter('green.lightness_h', 255)
        
        # Get parameters
        camera_topic = self.get_parameter('camera_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        stop_sign_path = self.get_parameter('stop_sign_path').value
        # Traffic light parameter values
        self.hue_red_l = self.get_parameter('red.hue_l').value
        self.hue_red_h = self.get_parameter('red.hue_h').value
        self.saturation_red_l = self.get_parameter('red.saturation_l').value
        self.saturation_red_h = self.get_parameter('red.saturation_h').value
        self.lightness_red_l = self.get_parameter('red.lightness_l').value
        self.lightness_red_h = self.get_parameter('red.lightness_h').value

        self.hue_yellow_l = self.get_parameter('yellow.hue_l').value
        self.hue_yellow_h = self.get_parameter('yellow.hue_h').value
        self.saturation_yellow_l = self.get_parameter('yellow.saturation_l').value
        self.saturation_yellow_h = self.get_parameter('yellow.saturation_h').value
        self.lightness_yellow_l = self.get_parameter('yellow.lightness_l').value
        self.lightness_yellow_h = self.get_parameter('yellow.lightness_h').value

        self.hue_green_l = self.get_parameter('green.hue_l').value
        self.hue_green_h = self.get_parameter('green.hue_h').value
        self.saturation_green_l = self.get_parameter('green.saturation_l').value
        self.saturation_green_h = self.get_parameter('green.saturation_h').value
        self.lightness_green_l = self.get_parameter('green.lightness_l').value
        self.lightness_green_h = self.get_parameter('green.lightness_h').value
        
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
        # Traffic light state counters to debounce detections
        self.red_count = 0
        self.yellow_count = 0
        self.green_count = 0
        self.traffic_state = 'unknown'  # 'red','yellow','green','unknown'
        self.traffic_confirm_threshold = 3
        # Red-stop timeout: if stopped for this many seconds while still seeing red, resume
        self.red_stop_since = None
        self.red_timeout_sec = 7.0

    def detect_traffic_light(self, frame_bgr):
        """Detect red/yellow/green traffic light in the image.
        Returns one of 'red', 'yellow', 'green', or None.
        Uses HSV color masks and simple contour area + ROI filtering.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        # Define ROIs: red -> top-right; yellow/green -> right-middle band
        # Red ROI: right 40% of image, top 33%
        red_x_start = int(width * 0.6)
        red_x_end = width
        red_y_start = 0
        red_y_end = height // 3
        roi_red = hsv[red_y_start:red_y_end, red_x_start:red_x_end]

        # Yellow/green ROI: right half, middle band
        yg_x_start = width // 2
        yg_x_end = width
        yg_y_start = height // 3
        yg_y_end = 2 * height // 3
        roi_yg = hsv[yg_y_start:yg_y_end, yg_x_start:yg_x_end]

        # Prepare masks for each ROI
        lower_red = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        upper_red = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])
        mask_red = cv2.inRange(roi_red, lower_red, upper_red)
        mask_red = cv2.GaussianBlur(mask_red, (5, 5), 0)

        lower_yellow = np.array([self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l])
        upper_yellow = np.array([self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h])
        mask_yellow = cv2.inRange(roi_yg, lower_yellow, upper_yellow)
        mask_yellow = cv2.GaussianBlur(mask_yellow, (5, 5), 0)

        lower_green = np.array([self.hue_green_l, self.saturation_green_l, self.lightness_green_l])
        upper_green = np.array([self.hue_green_h, self.saturation_green_h, self.lightness_green_h])
        mask_green = cv2.inRange(roi_yg, lower_green, upper_green)
        mask_green = cv2.GaussianBlur(mask_green, (5, 5), 0)

        # Helper to check if mask has a sufficiently large contour in ROI
        def mask_has_blob(mask, roi_x_offset, roi_y_offset, min_area=100):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    M = cv2.moments(cnt)
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10'] / M['m00']) + roi_x_offset
                    cy = int(M['m01'] / M['m00']) + roi_y_offset
                    return True, (cx, cy), area
            return False, None, 0

        red_ok, red_pt, red_area = mask_has_blob(mask_red, red_x_start, red_y_start)
        yellow_ok, yellow_pt, yellow_area = mask_has_blob(mask_yellow, yg_x_start, yg_y_start)
        green_ok, green_pt, green_area = mask_has_blob(mask_green, yg_x_start, yg_y_start)

        # Prefer the color with largest area if multiple detected
        candidates = []
        if red_ok:
            candidates.append(('red', red_area, red_pt))
        if yellow_ok:
            candidates.append(('yellow', yellow_area, yellow_pt))
        if green_ok:
            candidates.append(('green', green_area, green_pt))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][2]
    
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

            # Traffic light detection (debounced)
            tl_color, tl_pt = self.detect_traffic_light(frame)
            if tl_color == 'red':
                self.red_count += 1
                self.yellow_count = 0
                self.green_count = 0
            elif tl_color == 'yellow':
                self.yellow_count += 1
                self.red_count = 0
                self.green_count = 0
            elif tl_color == 'green':
                self.green_count += 1
                self.red_count = 0
                self.yellow_count = 0
            else:
                # if no detection, decay counts slowly
                self.red_count = max(0, self.red_count - 1)
                self.yellow_count = max(0, self.yellow_count - 1)
                self.green_count = max(0, self.green_count - 1)

            # Decide confirmed traffic state
            confirmed_state = None
            if self.red_count >= self.traffic_confirm_threshold:
                confirmed_state = 'red'
            elif self.yellow_count >= self.traffic_confirm_threshold:
                confirmed_state = 'yellow'
            elif self.green_count >= self.traffic_confirm_threshold:
                confirmed_state = 'green'

            if confirmed_state is not None:
                self.traffic_state = confirmed_state

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

            # Base linear velocity
            linear_vel = self.max_linear_vel

            # Modify linear velocity based on traffic light state
            if self.traffic_state == 'red':
                # mark time when robot first stopped for red
                if self.red_stop_since is None:
                    self.red_stop_since = time.time()
                elapsed = time.time() - self.red_stop_since
                # if red has persisted and we've been stationary for >= timeout, allow to resume slowly
                if elapsed >= self.red_timeout_sec:
                    linear_vel = max(0.05, 0.5 * self.max_linear_vel)
                    # log once when timeout triggers
                    if int(elapsed) == int(self.red_timeout_sec):
                        self.get_logger().info(f'Red persisted {elapsed:.1f}s — resuming movement')
                else:
                    linear_vel = 0.0
            elif self.traffic_state == 'yellow':
                # cancel red-stop timer
                self.red_stop_since = None
                linear_vel = max(0.05, 0.5 * self.max_linear_vel)
            elif self.traffic_state == 'green':
                self.red_stop_since = None
                linear_vel = self.max_linear_vel

            # Publish velocity command
            twist = Twist()
            twist.linear.x = float(linear_vel)
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

            # Display detected traffic light on debug image
            if tl_color is not None and tl_pt is not None:
                cv2.circle(debug_img, tl_pt, 8, (0, 0, 255) if tl_color == 'red' else (0, 255, 255) if tl_color == 'yellow' else (0, 255, 0), -1)
                cv2.putText(debug_img, tl_color.upper(), (tl_pt[0]+10, tl_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow('Canny Edges', edges)
        
        # Exit on ESC key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.get_logger().info('ESC pressed. Shutting down...')
            raise KeyboardInterrupt()


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
