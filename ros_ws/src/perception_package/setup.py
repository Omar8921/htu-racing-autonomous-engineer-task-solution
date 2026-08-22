from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'perception_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'models'),
            glob('models/*')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='omaralmadanii',
    maintainer_email='omaralmadani05@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = perception_package.camera_node:main',
            'lidar_node = perception_package.lidar_node:main',
            'visualization_node = perception_package.visualization_node:main'
        ],
    },
)
