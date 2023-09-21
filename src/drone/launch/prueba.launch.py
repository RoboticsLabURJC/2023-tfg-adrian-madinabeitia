#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # gazebo.launch.py package
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    # determine package where .world file is located
    share_pkg = get_package_share_directory('drone')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
          'world',
          default_value=[os.path.join(share_pkg, 'worlds', 'ocean_simulation.world'), ''],
          description='SDF world file'),
        gazebo
    ])