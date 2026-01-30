# Line Follower with Free Space Detection & Stop Sign Recognition

A simple ROS2 Python package for autonomous line following using Canny edge detection, free space detection, and stop sign recognition.

## Overview

The line follower uses:
1. **Canny Edge Detection** - to detect lane boundaries
2. **Free Space Detection** - to find the center of the free path
3. **Centroid Calculation** - to determine steering angle
4. **SIFT-based Stop Sign Detection** - to detect and stop at stop signs
5. **OpenCV Debug Visualization** - to display real-time processing results

## Features

- Subscribes to camera feed (default: `/camera/image_raw`)
- Applies Canny edge detection to find lane boundaries
- Detects free space and calculates steering angle
- Publishes velocity commands to `/cmd_vel`
- Shows 3 debug windows:
  - **Original**: Raw camera frame
  - **Canny Edges**: Detected edges from Canny filter
  - **Debug - Free Space**: Free space contours and centroid

## Installation

```bash
cd ~/turtlebot3_ws/src
# Package is already in line_follower_fsdetect/
cd ~/turtlebot3_ws
colcon build --packages-select line_follower_fsdetect
source install/setup.bash
```

## Usage

### Method 1: Using Launch File (Recommended)

```bash
ros2 launch line_follower_fsdetect line_follower.launch.py
```

### Method 2: Direct Node Execution

```bash
ros2 run line_follower_fsdetect line_follower
```

## Configuration

You can adjust parameters in the launch file or via command line:

```bash
ros2 launch line_follower_fsdetect line_follower.launch.py \
  -p camera_topic:=/camera/image_raw \
  -p cmd_vel_topic:=/cmd_vel \
  -p canny_threshold1:=50 \
  -p canny_threshold2:=150 \
  -p max_linear_vel:=0.2 \
  -p max_angular_vel:=1.0
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_topic` | `/camera/image_raw` | Input camera topic |
| `cmd_vel_topic` | `/cmd_vel` | Output velocity command topic |
| `canny_threshold1` | `50` | Canny lower threshold |
| `canny_threshold2` | `150` | Canny upper threshold |
| `max_linear_vel` | `0.2` | Maximum forward velocity (m/s) |
| `max_angular_vel` | `1.0` | Maximum rotation velocity (rad/s) |

## How It Works

1. **Image Capture**: Subscribes to camera feed
2. **Preprocessing**: Converts to grayscale
3. **Edge Detection**: Applies Canny edge detection
4. **Free Space Analysis**:
   - Scans horizontally at the bottom for lane boundaries
   - Vertically scans from bottom to find obstacles/edges
   - Calculates the centroid of the free space
5. **Steering Calculation**: Converts centroid position to steering angle
6. **Command Output**: Publishes Twist message for robot motion
7. **Visualization**: Displays debug images with `cv2.imshow()`

## Keyboard Controls

- **ESC**: Exit the program and shutdown node

## Troubleshooting

### No camera feed appearing
- Check if camera topic exists: `ros2 topic list | grep camera`
- Verify camera is publishing: `ros2 topic hz /camera/image_raw`

### Robot not moving
- Check `/cmd_vel` topic: `ros2 topic echo /cmd_vel`
- Verify max_linear_vel is > 0
- Check if robot's velocity controller is subscribed to `/cmd_vel`

### Poor line following performance
- Adjust Canny thresholds:
  - Lower `canny_threshold1` for more sensitive edge detection
  - Increase `canny_threshold2` for less noise
- Adjust `max_linear_vel` and `max_angular_vel`
- Ensure good lighting conditions for edge detection

## File Structure

```
line_follower_fsdetect/
├── line_follower_fsdetect/
│   ├── __init__.py
│   ├── freespace.py              # Free space detection algorithm
│   └── line_follower_node.py     # Main ROS2 node
├── launch/
│   └── line_follower.launch.py   # Launch file
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Dependencies

- `rclpy` - ROS2 Python client library
- `sensor_msgs` - ROS2 message types for sensors
- `geometry_msgs` - ROS2 message types for movement
- `cv_bridge` - Bridge between ROS and OpenCV
- `opencv-python` - Computer vision library

## License

Apache 2.0
