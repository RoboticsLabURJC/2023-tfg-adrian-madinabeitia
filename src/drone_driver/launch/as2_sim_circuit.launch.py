import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

namespace= "drone0"
sim_time = "true"
world = "/simple_circuit.world"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}

def generate_launch_description():
    sim_config = os.path.join(get_package_share_directory('drone_driver'), 'config')
    worlds_dir = os.path.join(get_package_share_directory('drone_driver'), 'worlds')
    tmux_yml = os.path.join(get_package_share_directory('drone_driver'), 'config/tmuxLaunch.yml')

    # Px4 autopilot gazebo launch
    gazeboPx4 = ExecuteProcess(
        cmd=[
            '/bin/bash', '-c',
            f'$AS2_GZ_ASSETS_SCRIPT_PATH/default_run.sh {sim_config}/world.json'
        ],
        name="gazebo",
        additional_env=env_vars
    )

    # Prepares the tmux session
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


    parseYaml = Node(
        package='drone_driver',
        executable='parseWorld.py',
        output='screen',
        arguments=[
            sim_config + '/world.json',
            worlds_dir + world,
            '53.462', '-10.734', '0.004', '-1.6'  # Drone coordinates
        ],
    )  

    return LaunchDescription([
        parseYaml,
        gazeboPx4,
        tmuxLauncher,
        tmuxAttach,
    ])