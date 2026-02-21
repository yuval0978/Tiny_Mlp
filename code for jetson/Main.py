"""
maze_solver_node.py

Single-file ROS2 (rclpy) program that:
 - subscribes to /scan (LaserScan) and /imu (sensor_msgs/Imu)
 - uses PID+TinyMLP to perform accurate 90° left/right turns
 - chooses turn direction when front < 0.10 m (choose side with larger clearance)
 - drives forward while stabilizing heading using IMU (yaw)
 - drives motors via Jetson.GPIO PWM (LN298 + GA25)

Before running:
  sudo apt install ros-humble-tf-transformations
  pip install Jetson.GPIO
Run as root (or with gpio permission).
Test with motors disconnected or very low PWM first!
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from tf_transformations import euler_from_quaternion

import Jetson.GPIO as GPIO
import threading
import numpy as np
import math
import time
import sys
import signal

# ------------------ Main ------------------
def main():
    rclpy.init()
    imu_node = ImuNode()
    lidar_node = LidarNode()
    pid = PIDController()
    mlp = TinyMLP(input_dim=10, hidden=48, out_dim=3)

    # spin ROS subscribers in background
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(imu_node)
    executor.add_node(lidar_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("starting maze solver")
    maze_solver(imu_node, lidar_node, pid, mlp)
    # run maze solver in main thread so we can catch KeyboardInterrupt
    '''try: stiil don't know if I want to do it this way.
        print("Starting maze solver (press Ctrl-C to stop).")
        maze_solver(imu_node, lidar_node, pid, mlp)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Shutting down: stopping motors, PWM, GPIO, rclpy.")
        stop_motors()
        for pwm in (pwm_left_fwd, pwm_left_bwd, pwm_right_fwd, pwm_right_bwd):
            try:
                pwm.stop()
            except Exception:
                pass
        GPIO.cleanup()
        rclpy.shutdown()'''


if __name__ == "__main__":
    main()
