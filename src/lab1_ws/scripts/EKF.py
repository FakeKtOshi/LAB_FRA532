#!/usr/bin/python3
from lab1_ws.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
import numpy as np
import math

class EKFNode(Node):
    def __init__(self):
        super().__init__('EKF_node')

        # Robot parameters
        self.wheel_radius = 0.1 
        self.wheel_base = 0.5    # distance between the centers of the wheels
        self.distance_wheel = 0.4 # distance between the wheels

        # State
        self.state = np.zeros(3) # [x, y, theta]

        # Normal parameters
        self.velocity = 0.0
        self.omega = 0.0
        self.last_time = None
        self.last_time_imu = None
        self.dt = 0.0
        self.dt_imu = 0.0   

        # EKF parameters
        self.P = np.identity(3)
        self.Q = np.eye(3) * 0.1
        self.R = np.eye(1) * 0.1

        # IMU parameters
        self.imu_omega = 0.0
        self.imu_theta = 0.0

        # Subscribers
        self.create_subscription(Imu, '/imu', self.imu_callback, 10) # IMU data
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # Publisher
        self.ekf_odom_publisher = self.create_publisher(Odometry, '/ekf_odom', 10)

    def joint_callback(self, msg):
        left_idx = msg.name.index('wheel_left_joint')
        right_idx = msg.name.index('wheel_right_joint')

        vel_left = msg.velocity[left_idx]
        vel_right = msg.velocity[right_idx]
        self.get_logger().info(f"Left: {vel_left}, Right: {vel_right}")

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = current_time
            return
        
        self.dt = current_time - self.last_time
        if self.dt > 1.0 or self.dt < 0.0:
            self.get_logger().warn("Time difference is too large or negative, skipping update.")
            return
        
        self.last_time = current_time

        # Linear velocity
        self.velocity = (vel_right + vel_left) * self.wheel_radius / 2.0 
        # Angular velocity
        self.omega = (vel_right - vel_left) * self.wheel_radius / self.distance_wheel 

        # Callout
        self.ekf_predict()
        stamp = msg.header.stamp
        self.publish_ekf_odometry(stamp)

    def imu_callback(self, msg):
        if not np.isfinite(msg.angular_velocity.z): # Prevent a function from crashing when the data is invalid or not a number
            self.get_logger().warn("IMU data is invalid!")
            return
        self.imu_omega = msg.angular_velocity.z
        
        current_time_imu  = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time_imu is None:
            self.last_time_imu = current_time_imu
            return
        
        self.dt_imu = current_time_imu - self.last_time_imu
        if self.dt_imu > 1.0 or self.dt_imu < 0.0:
            self.get_logger().warn("Time difference is too large or negative, skipping update.")
            return
        self.last_time_imu = current_time_imu

        self.imu_theta += self.imu_omega * self.dt_imu
        self.ekf_correct()
        self.get_logger().info(f"IMU theta received: {self.imu_theta}")

    def ekf_predict(self):
        # Predict State (Mean)
        x_new = self.state[0] + (self.velocity * math.cos(self.state[2]) * self.dt) # When we have the velocity and the angle, we can calculate the new position of the robot along with old one
        y_new = self.state[1] + (self.velocity * math.sin(self.state[2]) * self.dt)
        theta_new = self.state[2] + (self.omega * self.dt)

        # Predict State (Covariance)
        F = np.array([
            [1.0, 0.0, -self.velocity * math.sin(self.state[2]) * self.dt],
            [0.0, 1.0, self.velocity * math.cos(self.state[2]) * self.dt],
            [0.0, 0.0, 1.0]
        ])

        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = theta_new

        self.P = F @ self.P @ F.T + self.Q

    def ekf_correct(self):
        z = np.array([self.imu_theta]) # Theta
        #h = self.state[2]
        h = np.array([self.state[2]])
        H = np.array([[0.0, 0.0, 1.0]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        self.state = self.state + K @ (z - h)
        self.P = (np.eye(3) - K @ H) @ self.P

    def publish_ekf_odometry(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "ekf_odom"
        odom_msg.child_frame_id = "base_ekf"

        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.orientation.z = math.sin(self.state[2] / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.state[2] / 2.0)

        self.ekf_odom_publisher.publish(odom_msg)
        


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

