import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    kill_sesion = ExecuteProcess(
        cmd=['tmux', 'kill-session', '-t', 'drone0'],
        name="gazebo",
    )

    return LaunchDescription([
        kill_sesion,
    ])