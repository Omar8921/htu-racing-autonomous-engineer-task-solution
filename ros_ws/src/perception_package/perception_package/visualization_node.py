import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu
from custom_interfaces.msg import BoundingBoxArray

from cv_bridge import CvBridge
import cv2


"""
Node: Visualization

Inputs:
    /zed/left/image_rect_color
    /perception/yolo_bboxes
    /ground_truth/odom
    /imu/data

Output:
    /perception/debug_image
"""


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.declare_params()
        self.load_params()

        self.bridge = CvBridge()

        # Store images temporarily until the corresponding
        # BoundingBoxArray arrives from the camera node.
        self.image_cache = {}
        self.max_cached_images = 10

        # Vehicle state
        self.position = None
        self.yaw = None
        self.speed = None
        self.acceleration = None

        self.ID_TO_CLASS = {
            0: 'blue_cone',
            1: 'yellow_cone',
            2: 'orange_cone',
            3: 'large_orange_cone',
            4: 'unknown_cone'
        }

        self.ID_TO_BGR = {
            0: (255, 0, 0),      # blue
            1: (0, 255, 255),    # yellow
            2: (0, 165, 255),    # orange
            3: (0, 100, 255),    # large orange
            4: (128, 128, 128),  # unknown
        }

        self.create_subscribers()
        self.create_publishers()

        self.get_logger().info('Visualization Node started.')

    def declare_params(self):
        # Input topics
        self.declare_parameter(
            'image_input_topic',
            '/zed/left/image_rect_color'
        )

        self.declare_parameter(
            'bboxes_input_topic',
            '/perception/yolo_bboxes'
        )

        self.declare_parameter(
            'odom_input_topic',
            '/ground_truth/odom'
        )

        self.declare_parameter(
            'imu_input_topic',
            '/imu/data'
        )

        # Output topics
        self.declare_parameter(
            'debug_image_output_topic',
            '/perception/debug_image'
        )

        # General
        self.declare_parameter('verbose', False)

    def load_params(self):
        self.image_input_topic = str(
            self.get_parameter('image_input_topic').value
        )

        self.bboxes_input_topic = str(
            self.get_parameter('bboxes_input_topic').value
        )

        self.odom_input_topic = str(
            self.get_parameter('odom_input_topic').value
        )

        self.imu_input_topic = str(
            self.get_parameter('imu_input_topic').value
        )

        self.debug_image_output_topic = str(
            self.get_parameter('debug_image_output_topic').value
        )

        self.verbose = bool(
            self.get_parameter('verbose').value
        )

    def create_subscribers(self):
        self.image_sub = self.create_subscription(
            Image,
            self.image_input_topic,
            self.image_callback,
            10
        )

        self.bboxes_sub = self.create_subscription(
            BoundingBoxArray,
            self.bboxes_input_topic,
            self.bboxes_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_input_topic,
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_input_topic,
            self.imu_callback,
            10
        )

    def create_publishers(self):
        self.debug_image_pub = self.create_publisher(
            Image,
            self.debug_image_output_topic,
            10
        )

    def image_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        stamp = self.stamp_to_ns(msg.header.stamp)

        self.image_cache[stamp] = image

        # Prevent the cache from growing forever
        while len(self.image_cache) > self.max_cached_images:
            oldest_stamp = next(iter(self.image_cache))
            del self.image_cache[oldest_stamp]

    def bboxes_callback(self, msg: BoundingBoxArray):
        stamp = self.stamp_to_ns(msg.header.stamp)

        # Get the exact image used by the camera node
        if stamp not in self.image_cache:
            if self.verbose:
                self.get_logger().warn(
                    'Could not find matching image for bounding boxes.'
                )
            return

        image = self.image_cache.pop(stamp)

        # Draw detections
        for bbox in msg.boxes:
            class_id = bbox.class_id

            name = self.ID_TO_CLASS.get(
                class_id,
                f'class_{class_id}'
            )

            color = self.ID_TO_BGR.get(
                class_id,
                (255, 255, 255)
            )

            x1 = int(bbox.x1)
            y1 = int(bbox.y1)
            x2 = int(bbox.x2)
            y2 = int(bbox.y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image,
                f'{name}: {bbox.confidence:.2f}',
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Draw vehicle information
        self.draw_vehicle_state(image)

        # Convert back to ROS Image
        debug_msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding='bgr8'
        )

        debug_msg.header.stamp = msg.header.stamp
        debug_msg.header.frame_id = msg.header.frame_id

        self.debug_image_pub.publish(debug_msg)

    def odom_callback(self, msg: Odometry):
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        # Quaternion -> yaw
        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y ** 2 +
            q.z ** 2
        )

        self.yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )

        velocity = msg.twist.twist.linear

        self.speed = math.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2 +
            velocity.z ** 2
        )

    def imu_callback(self, msg: Imu):
        self.acceleration = msg.linear_acceleration.x

    def draw_vehicle_state(self, image):
        lines = []

        if self.position is not None:
            lines.append(
                f'Position: '
                f'({self.position[0]:.2f}, '
                f'{self.position[1]:.2f}) m'
            )

        if self.yaw is not None:
            lines.append(
                f'Orientation: '
                f'{math.degrees(self.yaw):.2f} deg'
            )

        if self.speed is not None:
            lines.append(
                f'Speed: {self.speed:.2f} m/s'
            )

        if self.acceleration is not None:
            lines.append(
                f'Acceleration: '
                f'{self.acceleration:.2f} m/s^2'
            )

        x = 30
        y = 40
        line_spacing = 35

        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (x, y + i * line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    @staticmethod
    def stamp_to_ns(stamp):
        return (
            stamp.sec * 1_000_000_000 +
            stamp.nanosec
        )


def main():
    rclpy.init()

    node = VisualizationNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()