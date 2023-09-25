"""Simulate a Tello drone"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

ns = 'tello_drone'
def generate_launch_description():
    world_path = os.path.join(get_package_share_directory('drone_driver'), 'worlds', 'ocean_simulation.world')
    urdf_path = os.path.join(get_package_share_directory('drone_driver'), 'urdf', 'tello.urdf')

    return LaunchDescription([

        # Spawn tello.urdf
        Node(package='drone_driver', executable='inject_entity.py', output='screen',
            arguments=[urdf_path, '0', '0', '2', '0']),
        
        # Publish static transforms
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             arguments=[urdf_path]),

        # Joystick driver, generates /namespace/joy messages
        Node(package='joy', executable='joy_node', output='screen',
             namespace=ns),

        # Launch Gazebo, loading tello.world
        ExecuteProcess(cmd=[
            'gazebo',
            '--verbose',
            '-s', 'libgazebo_ros_init.so',  # Publish /clock
            '-s', 'libgazebo_ros_factory.so',  # Provide gazebo_ros::Node
            world_path
        ], output='screen'),
    ])
