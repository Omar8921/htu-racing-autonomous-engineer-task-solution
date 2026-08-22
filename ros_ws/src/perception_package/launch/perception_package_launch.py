from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    package_share = get_package_share_directory('perception_package')

    camera_node = Node(
        package='perception_package',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    lidar_node = Node(
        package='perception_package',
        executable='lidar_node',
        name='lidar_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    visualization_node = Node(
        package='perception_package',
        executable='visualization_node',
        name='visualization_node',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([
        camera_node,
        lidar_node,
        visualization_node
    ])