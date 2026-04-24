#!/usr/bin/python3

from lab1_ws.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import numpy as np

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class DifferentialDriveNode(Node):
    def __init__(self):
        super().__init__('differential_drive_node')

        # Robot parameters
        self.wheel_radius = 0.1 
        self.wheel_base = 0.5    # distance between the centers of the wheels
        self.distance_wheel = 0.4 # distance between the wheels

        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self.state = np.zeros(3) # [x, y, theta]
        #self.initial_time = self.get_clock().now()
        self.last_time = None
        # self.x_initial = 0
        # self.y_initial = 0
        # self.theta_initial = 0

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10) # Wheel data

        # Publisher
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10)
        #self.state_publisher = self.create_publisher(Float64, '/state', 10)

    def joint_callback(self, msg):
        # vel_left = msg.velocity[0]
        # vel_right = msg.velocity[1]
        
        left_idx = msg.name.index('wheel_left_joint')
        right_idx = msg.name.index('wheel_right_joint')

        vel_left = msg.velocity[left_idx]
        vel_right = msg.velocity[right_idx]
        self.get_logger().info(f"Left: {vel_left}, Right: {vel_right}")

        # current_time = self.get_clock().now()
        # if self.last_time is None:
        #     self.last_time = current_time
        #     return
        
        # dt = (current_time - self.last_time).nanoseconds / 1e9 # Convert a time to seconds
        # self.last_time = current_time

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = current_time
            return
        
        dt = current_time - self.last_time
        if dt > 1.0 or dt < 0.0:
            self.get_logger().warn("Invalid dt time")
            return

        self.last_time = current_time

        # Linear velocity
        self.velocity = (vel_right + vel_left) * self.wheel_radius / 2.0 
        # Angular velocity
        self.omega = (vel_right - vel_left) * self.wheel_radius / self.distance_wheel 
        
        x_new = self.state[0] + (self.velocity * math.cos(self.state[2]) * dt) # When we have the velocity and the angle, we can calculate the new position of the robot along with old one
        y_new = self.state[1] + (self.velocity * math.sin(self.state[2]) * dt)
        theta_new = self.state[2] + (self.omega * dt)

        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = theta_new

        stamp = msg.header.stamp # Use for a timestamp for the odometry message
        self.publish_odometry(stamp)
        # self.publish_state(self.state)

    def publish_odometry(self, stamp):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        odom_msg.pose.pose.position.z = 0.0 # Height is zero because we are in 2D plane
        
        # Convert theta to quaternion
        odom_msg.pose.pose.orientation.z = math.sin(self.state[2] / 2.0) 
        odom_msg.pose.pose.orientation.w = math.cos(self.state[2] / 2.0)

        odom_msg.twist.twist.linear.x = self.velocity
        odom_msg.twist.twist.angular.z = self.omega

        self.odom_publisher.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self.state[2] / 2.0)
        t.transform.rotation.w = math.cos(self.state[2] / 2.0)
        self.tf_broadcaster.sendTransform(t)
    
    # def publish_state(self, state):
    #     self.state_publisher.publish(state)

def main(args=None):
    rclpy.init(args=args)
    node = DifferentialDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
