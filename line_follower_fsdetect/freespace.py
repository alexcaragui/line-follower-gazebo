import cv2
import numpy as np
import math


def get_freespace_angle(frame, img_debug=None):
    """
    Calculate steering angle from free space in binary/edge-detected frame.
    
    Args:
        frame: Binary input image (from Canny edge detection or thresholding)
        img_debug: Optional BGR image to draw debug visualizations on
        
    Returns:
        angle_deg: Steering angle in degrees
        centroid: (x, y) tuple of the free space centroid
    """
    height, width = frame.shape[:2]
    height_idx = height - 1
    width_idx = width - 1
    
    center_x = width // 2
    base_y = height_idx - 10
    
    dreapta_lim = center_x  # Right limit
    stanga_lim = center_x   # Left limit

    # 1. Find Right Limit
    for i in range(center_x, width):
        if frame[base_y, i] > 0:
            dreapta_lim = i
            break
            
    # 2. Find Left Limit
    for i in range(0, center_x):
        if frame[base_y, center_x - i] > 0:
            stanga_lim = center_x - i
            break

    # Boundary safety
    if stanga_lim == center_x:
        stanga_lim = 1
    if dreapta_lim == center_x:
        dreapta_lim = width_idx

    contur = []
    contur.append((stanga_lim, base_y))

    if img_debug is not None:
        cv2.circle(img_debug, (dreapta_lim, base_y), 5, (0, 0, 255), -1)
        cv2.circle(img_debug, (stanga_lim, base_y), 5, (0, 0, 255), -1)

    # 3. Column Scanning (Vertical search for free space)
    # Scanning from Left Limit to Right Limit in steps of 10
    for j in range(stanga_lim, dreapta_lim + 1, 10):
        found_obstacle = False
        for i in range(base_y, 9, -1):
            if frame[i, j] > 0:
                if img_debug is not None:
                    cv2.line(img_debug, (j, base_y), (j, i), (255, 0, 0), 2)
                contur.append((j, i))
                found_obstacle = True
                break
        
        # If no obstacle found up to row 10
        if not found_obstacle:
            contur.append((j, 10))
            if img_debug is not None:
                cv2.line(img_debug, (j, base_y), (j, 10), (255, 0, 0), 2)

    contur.append((dreapta_lim, base_y))
    
    # 4. Calculate Centroid of the free area
    contur_np = np.array(contur, dtype=np.int32)
    moments = cv2.moments(contur_np)
    
    if moments["m00"] != 0:
        x_mean = int(moments["m10"] / moments["m00"])
        y_mean = int(moments["m01"] / moments["m00"])
    else:
        x_mean, y_mean = center_x, base_y

    # Draw centroid on debug image
    if img_debug is not None:
        cv2.circle(img_debug, (x_mean, y_mean), 8, (0, 255, 0), -1)
        cv2.line(img_debug, (center_x, height_idx), (x_mean, y_mean), (0, 255, 0), 2)

    # 5. Calculate Steering Angle
    # atan2(y_diff, x_diff) -> converted to degrees
    # The +1.57 (π/2) aligns the coordinate system so 0 is straight ahead
    angle_rad = (math.atan2(y_mean - height_idx, x_mean - center_x) + 1.57)
    angle_deg = angle_rad * 57.2958  # (180/pi)
    
    return angle_deg, (x_mean, y_mean)
