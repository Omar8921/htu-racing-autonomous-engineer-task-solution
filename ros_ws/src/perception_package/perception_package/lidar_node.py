import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener, TransformException

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import numpy as np
from core.ground_removal import remove_ground
from core.clustering import cluster_points

"""
Node: LiDAR Perception

Input:
    /velodyne_points

Outputs:
    /perception/roi_points
    /perception/non_ground_points
    /perception/clustered_points

Coordinate Frames:
    Source:
        The frame provided by the incoming PointCloud2 message
        (msg.header.frame_id).

    Target:
        base_footprint

The LiDAR points should be transformed from the sensor frame into
base_footprint before processing so that the coordinates are aligned
with the vehicle.

If this is unfamiliar, briefly read about ROS 2 TF2 and coordinate
frame transformations.
"""

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.declare_params()
        self.load_params()

        self.initialize_storage()

        self.create_subscribers()
        self.create_publishers()

        self.get_logger().info('LiDAR Node started.')

    def declare_params(self):
        # Input Topics
        self.declare_parameter('points_input_topic', '/velodyne_points')

        # Output Topics
        self.declare_parameter('non_ground_output_topic', '/perception/non_ground_points')
        self.declare_parameter('roi_points_output_topic', '/perception/roi_points')
        self.declare_parameter('clustered_points_output_topic', '/perception/clustered_points')

        # Frame Params
        self.declare_parameter('source_frame', 'velodyne')
        self.declare_parameter('target_frame', 'base_footprint')

        # ROI Params
        self.declare_parameter('x_min', 0.0)
        self.declare_parameter('x_max', 16.0)
        self.declare_parameter('y_min', -6.0)
        self.declare_parameter('y_max', 6.0)
        self.declare_parameter('z_min', -1.0)
        self.declare_parameter('z_max', 1.0)

        # General Params
        self.declare_parameter('verbose', False)
        self.declare_parameter('visualize', True)

    def load_params(self):
        # Input Topics
        self.points_input_topic = str(self.get_parameter('points_input_topic').value)

        # Output Topics
        self.non_ground_output_topic = str(self.get_parameter('non_ground_output_topic').value)
        self.roi_points_output_topic = str(self.get_parameter('roi_points_output_topic').value)
        self.clustered_points_output_topic = str(self.get_parameter('clustered_points_output_topic').value)

        # Frame Params
        self.source_frame = str(self.get_parameter('source_frame').value)
        self.target_frame = str(self.get_parameter('target_frame').value)

        # ROI Params
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.y_min = float(self.get_parameter('y_min').value)
        self.y_max = float(self.get_parameter('y_max').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)

        # General Params
        self.verbose = bool(self.get_parameter('verbose').value)
        self.visualize = bool(self.get_parameter('visualize').value)

    def initialize_storage(self):
        self.tf_buffer = Buffer(cache_time=Duration(seconds=1))
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def create_subscribers(self):
        self.input_points_sub = self.create_subscription(
            PointCloud2, self.points_input_topic, self.points_callback, 10
        )

    def create_publishers(self):
        self.non_ground_pub = self.create_publisher(
            PointCloud2, self.non_ground_output_topic, 10
        )

        self.roi_points_pub = self.create_publisher(
            PointCloud2, self.roi_points_output_topic, 10
        )

        self.clustered_points_pub = self.create_publisher(
            PointCloud2, self.clustered_points_output_topic, 10
        )

    def points_callback(self, msg: PointCloud2):
        # For points publishing purposes
        stamp = msg.header.stamp

        # Convert msg to NumPy array
        points = point_cloud2.read_points_numpy(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        )

        if self.verbose:
            self.get_logger().info(f'Number of detected LiDAR points: {points.shape[0]}')

        # Transform points to target frame
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.target_frame,
                source_frame=self.source_frame,
                time=Time(seconds=0.05)
            )
        except TransformException as e:
            self.get_logger().warn(f"Could not look up transform: {e}")

        t = transform.transform.translation
        translation = np.array([
            t.x,
            t.y,
            t.z
        ], dtype=np.float32)

        q = transform.transform.rotation
        qx = q.x
        qy = q.y
        qz = q.z
        qw = q.w 
        rotation_matrix = np.array([
            [
                1 - 2*(qy**2 + qz**2),
                2*(qx*qy - qz*qw),
                2*(qx*qz + qy*qw)
            ],
            [
                2*(qx*qy + qz*qw),
                1 - 2*(qx**2 + qz**2),
                2*(qy*qz - qx*qw)
            ],
            [
                2*(qx*qz - qy*qw),
                2*(qy*qz + qx*qw),
                1 - 2*(qx**2 + qy**2)
            ]
        ])

        transformed_points = points @ rotation_matrix.T + translation

        if self.verbose:
            self.get_logger().info(
                f'Successfully transformed points to target frame. ' + 
                f'Number of LiDAR points: {transformed_points.shape[0]}')

        # ROI Filtering
        filtered_points = transformed_points[
            (self.x_min <= transformed_points[:, 0]) &
            (transformed_points[:, 0] <= self.x_max) &

            (self.y_min <= transformed_points[:, 1]) &
            (transformed_points[:, 1] <= self.y_max) &

            (self.z_min <= transformed_points[:, 2]) &
            (transformed_points[:, 2] <= self.z_max)
        ]

        if self.verbose:
            self.get_logger().info(
                f'Number of points after ROI filtering: {filtered_points.shape[0]}'
            )
        if self.visualize:
            roi_msg = self.numpy_to_pointcloud2(filtered_points, stamp)
            self.roi_points_pub.publish(roi_msg)

        # Ground Removal
        non_ground_points = remove_ground(filtered_points, 100, 0.10)

        if self.verbose:
            self.get_logger().info(
                f'Number of non-ground points: {non_ground_points.shape[0]}'
            )
        if self.visualize:
            non_ground_msg = self.numpy_to_pointcloud2(non_ground_points, stamp)
            self.non_ground_pub.publish(non_ground_msg)

        # Clustering
        clusters = cluster_points(non_ground_points, 0.1, 2)
        
        if self.verbose:
            self.get_logger().info(
                f'Number of clusters: {clusters.shape[0]}'
            )
        if self.visualize:
            clusters_msg = self.numpy_to_pointcloud2(clusters, stamp)
            self.clustered_points_pub.publish(clusters_msg)
            

    def numpy_to_pointcloud2(self, points: np.ndarray, stamp):
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        header = Header()
        header.stamp = stamp
        header.frame_id = self.target_frame

        return point_cloud2.create_cloud_xyz32(
            header, points
        )

            

def main():
    rclpy.init()
    node = LidarNode()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()    


if __name__ == '__main__':
    main()