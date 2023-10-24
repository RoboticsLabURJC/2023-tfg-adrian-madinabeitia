import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

namespace= "drone0"
sim_time = "true"

def generate_launch_description():
    sim_config = os.path.join(get_package_share_directory('drone_driver'), 'config')
    worlds = os.path.join(get_package_share_directory('drone_driver'), 'worlds')
    tmux_yml = os.path.join(get_package_share_directory('drone_driver'), 'config/tmuxLaunch.yml')

    gazeboPx4 = ExecuteProcess(
        cmd=[
            '/bin/bash', '-c',
            f'$AS2_GZ_ASSETS_SCRIPT_PATH/default_run.sh {sim_config}/world.json'
        ],
        name="gazebo",
    )

    tmuxLauncher = ExecuteProcess(
        cmd=['tmuxinator', 'start', '-n', namespace, '-p', tmux_yml, 
             "drone_namespace=" + namespace, 
             "simulation_time=" + sim_time,
             "config_path=" + sim_config],

        name="tmuxLauncher",
        output='screen'
    )

    tmuxAttach = ExecuteProcess(
        cmd=['gnome-terminal', '--', 'tmux', 'attach-session', '-t', namespace],
        name="attach",
    )


    set_world_path_process = ExecuteProcess(
        cmd=[
            '/usr/bin/env', 'python3',
            sim_config + '/parse.py',
            sim_config + '/world.json',
            worlds + '/ocean_simulation.world'
        ],
        name='set_world_path_process',
        output='screen',
        additional_env={'WORLD_PATH': sim_config+ "world.json"},
    )

    return LaunchDescription([
        set_world_path_process,
        gazeboPx4,
        tmuxLauncher,
        tmuxAttach,
    ])