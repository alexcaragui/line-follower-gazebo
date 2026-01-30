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
        
        # Right-arrow sign detection (SIFT + FLANN)
        self.right_arrow_template = None
        self.right_arrow_kp = None
        self.right_arrow_des = None
        self.right_arrow_detected = False
        self.turn_right_counter = 0
        self.turn_right_threshold = 3  # confirm detection over N frames
        
        # FLANN matcher for arrow sign detection
        FLANN_INDEX_KDTREE = 0
        index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}
        search_params = {'checks': 50}
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Sign detection thresholds
        self.MIN_MATCH_COUNT = 5
        self.MIN_MSE_DECISION = 70000
        # Right-arrow timing: wait before turning and cooldown
        self.right_detected_since = None
        self.right_wait_sec = 3.0
        self.right_cooldown_until = 0.0
        # Executing turn state (sustain turning for a duration)
        self.executing_turn_until = 0.0
        self.turn_duration = 1.2  # seconds to perform the turn once started
        self.turn_angular_speed = 1.0  # angular speed used during turn
        
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
        
        # Load right arrow template
        pkg_dir = '/home/alex/turtlebot3_ws/src/line_follower_fsdetect'
        right_arrow_path = os.path.join(pkg_dir, 'right.png')
        if os.path.exists(right_arrow_path):
            self.right_arrow_template = cv2.imread(right_arrow_path, 0)  # grayscale
            if self.right_arrow_template is not None:
                self.right_arrow_kp, self.right_arrow_des = self.sift.detectAndCompute(
                    self.right_arrow_template, None
                )
                self.get_logger().info(f'Loaded right arrow template from {right_arrow_path}')
                self.get_logger().info(f'Right arrow has {len(self.right_arrow_kp)} SIFT keypoints')
            else:
                self.get_logger().warn(f'Failed to load right arrow from {right_arrow_path}')
        else:
            self.get_logger().info(f'Right arrow template not found at {right_arrow_path}')
        
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
    
    def detect_right_arrow_sign(self, frame_gray):
        """Detect right-arrow sign using SIFT+FLANN matching.
        Returns True if detected with sufficient matches and low MSE, False otherwise."""
        if self.right_arrow_des is None or len(self.right_arrow_kp) < self.MIN_MATCH_COUNT:
            return False
        
        try:
            kp_frame, des_frame = self.sift.detectAndCompute(frame_gray, None)
            
            if des_frame is None or len(kp_frame) < self.MIN_MATCH_COUNT:
                return False
            
            # Use FLANN matcher
            matches = self.flann_matcher.knnMatch(des_frame, self.right_arrow_des, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Check if enough matches
            if len(good_matches) < self.MIN_MATCH_COUNT:
                return False
            
            # Compute homography and MSE to verify match quality
            try:
                src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.right_arrow_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Calculate MSE
                mse = np.sum((src_pts - dst_pts) ** 2) / len(src_pts)
                
                if mse < self.MIN_MSE_DECISION:
                    self.get_logger().info(f'Right arrow detected! MSE: {mse:.2f}, Matches: {len(good_matches)}')
                    return True
            except Exception as e:
                self.get_logger().warn(f'Homography error in right arrow detection: {e}')
                return False
            
            return False
        except Exception as e:
            self.get_logger().warn(f'Error in right arrow SIFT detection: {e}')
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

            # Detect right-arrow sign (debounced) with cooldown
            right_arrow_detected = self.detect_right_arrow_sign(gray)
            # ignore detection if in cooldown period
            if time.time() < self.right_cooldown_until:
                right_arrow_detected = False
            if right_arrow_detected:
                self.turn_right_counter += 1
            else:
                self.turn_right_counter = max(0, self.turn_right_counter - 1)

            # Confirm right arrow detection over threshold frames
            if self.turn_right_counter >= self.turn_right_threshold:
                if not self.right_arrow_detected:
                    # start waiting timer before turning
                    self.right_arrow_detected = True
                    self.right_detected_since = time.time()
            elif self.turn_right_counter == 0:
                self.right_arrow_detected = False
                self.right_detected_since = None

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

            # If right arrow detected, wait then start a sustained turn
            if self.right_arrow_detected:
                # if still in cooldown, ignore detection
                if time.time() < self.right_cooldown_until:
                    # do nothing special, proceed with normal steering
                    pass
                else:
                    # ensure we have a start time for waiting
                    if self.right_detected_since is None:
                        self.right_detected_since = time.time()
                    elapsed = time.time() - self.right_detected_since
                    if elapsed < self.right_wait_sec:
                        # show waiting countdown on debug image while still moving towards sign
                        cv2.putText(debug_img, f'Approaching then turn: {self.right_wait_sec - elapsed:.1f}s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        # start the sustained turn period
                        if time.time() >= self.executing_turn_until:
                            self.executing_turn_until = time.time() + self.turn_duration
                            self.get_logger().info('Starting sustained right turn')
                            # set a cooldown to avoid immediate re-trigger after the turn
                            self.right_cooldown_until = time.time() + 4.0
                            # reset detection counters
                            self.right_arrow_detected = False
                            self.turn_right_counter = 0
                            self.right_detected_since = None
            # Modify linear velocity based on traffic light state
            elif self.traffic_state == 'red':
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

            # If currently executing a sustained turn, override steering for the duration
            if time.time() < self.executing_turn_until:
                # sustain a right turn: negative angular for right (node uses -angular_vel)
                angular_vel = self.turn_angular_speed
                linear_vel = 0.1
            else:
                # clear executing flag when done
                if self.executing_turn_until != 0.0 and time.time() >= self.executing_turn_until:
                    self.executing_turn_until = 0.0

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

            # Display right arrow detection status
            if self.right_arrow_detected:
                cv2.putText(
                    debug_img,
                    'RIGHT ARROW DETECTED!',
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(debug_img, (5, 100), (debug_img.shape[1]-5, 125), (0, 0, 255), 2)

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
