#!/usr/bin/python3

from lab1_ws.dummy_module import dummy_function, dummy_var
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy   

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from scipy.spatial import KDTree

class ICPNode(Node):
    def __init__(self):
        super().__init__('icp_node')

        self.tf_broadcaster = TransformBroadcaster(self)

        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        depth=10)

        # Subscribe
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile=qos_profile)
        self.create_subscription(Odometry, '/ekf_odom', self.ekf_odom_callback, 10)

        # Publisher
        self.icp_odom_publisher = self.create_publisher(Odometry, '/icp_odom', 10)

        # Variables
        self.prev_scan = None
        self.curr_scan = None
        self.ekf_pose = np.zeros(3) # [x, y, theta]
        self.pose = np.zeros(3) # [x, y, theta]

        # self.R = np.eye(2) # Rotation matrix
        # self.t = np.zeros(2) # Translation vector

    def ekf_odom_callback(self, msg):
        self.ekf_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y,
                                  2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)])

    def scan2points(self, msg):
        # Convert laser scan to points
        ranges = []

        for i in range(len(msg.ranges)):
            if msg.ranges[i] < msg.range_max and msg.ranges[i] > msg.range_min:
                angle = msg.angle_min + i * msg.angle_increment
                x = msg.ranges[i] * np.cos(angle)
                y = msg.ranges[i] * np.sin(angle)

                ranges.append([x, y])
        return np.array(ranges)
    
    # def find_nearest(self, source, destination):
    #     indices = []
    #     distances = []
        
    #     for si in source:
    #         min_dist = float('inf')
    #         min_idx = 0
            
    #         for j, dj in enumerate(destination):
    #             # paper line 5: ‖(dⱼ, sᵢ)‖²
    #             dist = np.sum((si - dj)**2)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 min_idx = j
            
    #         indices.append(min_idx)
    #         distances.append(np.sqrt(min_dist))
        
    #     return np.array(distances), np.array(indices)

    def find_nearest(self, source, destination):
        tree = KDTree(destination)
        distances, indices = tree.query(source)
        return np.array(distances), np.array(indices)
    
    def icp(self, source, destination):
        sur = source
        des = destination
        
        max_iterations = 5
        error = float('inf') 
        prev_error = float('inf')
        threshold = 0.001
        
        for i in range(max_iterations):
            # Find the nearest neighbors
            distances, indices = self.find_nearest(sur, des)
            matched_des = des[indices]

            # Compute the centroids of the source and matched reference points
            centroid_sur = np.mean(sur, axis=0)
            centroid_des = np.mean(matched_des, axis=0)

            # Center the points
            centered_sur = sur - centroid_sur
            centered_des = matched_des - centroid_des

            # Compute the covariance matrix
            H = np.dot(centered_sur.T, centered_des)

            # Singular Value Decomposition (SVD)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # Ensure a proper rotation (det(R) should be 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)

            T = centroid_des - np.dot(R, centroid_sur)

            # Transform the source points
            sur = np.dot(R, sur.T).T + T

            # Compute the mean error
            error = np.mean(distances)

            if prev_error - error < threshold:
                break
            prev_error = error

            self.get_logger().info(f"ICP Iteration {i+1}: Error = {error:.4f}")

        return R, T
    
    def publish_icp_odom(self, msg):
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = "icp_odom"
        odom_msg.child_frame_id = "base_link"

        # Populate the odometry message with the ICP pose
        odom_msg.pose.pose.position.x = self.pose[0]
        odom_msg.pose.pose.position.y = self.pose[1]
        odom_msg.pose.pose.position.z = 0.0
        
        odom_msg.pose.pose.orientation.z = np.sin(self.pose[2] / 2.0)
        odom_msg.pose.pose.orientation.w = np.cos(self.pose[2] / 2.0)

        self.icp_odom_publisher.publish(odom_msg)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp  
        t.header.frame_id = 'icp_odom'    
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.pose[0]
        t.transform.translation.y = self.pose[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self.pose[2] / 2.0)
        t.transform.rotation.w = math.cos(self.pose[2] / 2.0)
        self.tf_broadcaster.sendTransform(t)
            
    def scan_callback(self, msg):
        self.curr_scan = self.scan2points(msg)
        if self.prev_scan is not None:
            self.R, self.T = self.icp(self.curr_scan, self.prev_scan)

            dtheta = np.arctan2(self.R[1, 0], self.R[0, 0])
            dx = self.T[0]
            dy = self.T[1]

            #Update the pose
            # self.pose[0] += self.T[0]
            # self.pose[1] += self.T[1]
            # self.pose[2] += np.arctan2(self.R[1, 0], self.R[0, 0])

            self.pose[0] = self.ekf_pose[0] + dx   # ← use ekf as base!
            self.pose[1] = self.ekf_pose[1] + dy
            self.pose[2] = self.ekf_pose[2] + dtheta
            self.publish_icp_odom(msg)
        self.prev_scan = self.curr_scan

def main(args=None):
    rclpy.init(args=args)
    node = ICPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
