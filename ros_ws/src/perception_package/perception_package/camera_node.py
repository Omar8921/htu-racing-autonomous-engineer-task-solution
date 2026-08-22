import rclpy
from rclpy.node import Node

from pathlib import Path
from ament_index_python.packages import get_package_share_directory


from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu
from custom_interfaces.msg import BoundingBox, BoundingBoxArray

import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge

"""
Node: Camera Perception

Input:
    /zed/left/image_rect_color

Output:
    /perception/yolo_bboxes
"""

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Create and load ROS2 params in an abstract manner
        self.declare_params()
        self.load_params()

        self.get_logger().info('Creating bridge...')
        self.bridge = CvBridge()

        # Instantiate a YOLO model from model path
        self.get_logger().info('Loading YOLO...')
        self.model = YOLO(self.model_path, task='detect')
        self.get_logger().info('YOLO loaded.')
        self.get_logger().info(f'YOLO Labels: {self.model.names}')

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

        # Create subscribers and publishers in an abstract manner
        self.create_subscribers()
        self.create_publishers()

        # Alert that node started
        self.get_logger().info('Camera Node started.')

    def declare_params(self):
        package_share = Path(
            get_package_share_directory('perception_package')
        )
        default_model_path = package_share / 'models' / 'yolov26.pt'

        # Topic Params
        self.declare_parameter('image_input_topic', '/zed/left/image_rect_color')
        self.declare_parameter('bboxes_output_topic', '/perception/yolo_bboxes')

        # YOLO Params
        self.declare_parameter('model_path', str(default_model_path))

        # General Params
        self.declare_parameter('verbose', False)

    def load_params(self):
        # Topic Params
        self.image_input_topic = str(self.get_parameter('image_input_topic').value)
        self.bboxes_output_topic = str(self.get_parameter('bboxes_output_topic').value)

        # YOLO Params
        self.model_path = str(self.get_parameter('model_path').value)

        # General Params
        self.verbose = bool(self.get_parameter('verbose').value)

    def create_subscribers(self):
        self.input_image_sub = self.create_subscription(
            Image, self.image_input_topic, self.image_callback, 10
        )

    def create_publishers(self):
        self.bboxes_pub = self.create_publisher(
            BoundingBoxArray, self.bboxes_output_topic, 10
        )

    def image_callback(self, msg: Image):
        '''
        This callback method receives msg, converts it to numpy array, 
        runs YOLO model, and extracts bounding boxes.
        '''
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # Run YOLO inference on image
        results = self.model(image)
        bbox_array = BoundingBoxArray()
        bbox_array.header.stamp = msg.header.stamp
        bbox_array.header.frame_id = msg.header.frame_id

        # Loop over results
        for result in results:
            # Extract bounding boxes containing detections
            boxes = result.boxes

            # Loop over each box
            for box in boxes:
                # Extract bounding box coordinates in xyxy format (top-left, bottom-right)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())

                # Store detected object data to a message
                bbox = BoundingBox()
                bbox.class_id = class_id
                bbox.confidence = confidence

                bbox.x1 = x1
                bbox.y1 = y1
                bbox.x2 = x2
                bbox.y2 = y2

                name = self.ID_TO_CLASS[class_id]

                if self.verbose:
                    self.get_logger().info(
                        f"Detected {name} | "
                        f"confidence={confidence:.2f} | "
                        f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                    )

               # Append to bounding box array
                bbox_array.boxes.append(bbox)

        # Publish bounding boxes array
        self.bboxes_pub.publish(bbox_array)


def main():
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()