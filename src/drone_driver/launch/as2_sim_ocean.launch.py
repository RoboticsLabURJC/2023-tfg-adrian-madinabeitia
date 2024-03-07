import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

namespace= "drone0"
sim_time = "true"
world = "/ocean.world"

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
            '0.0', '0.0', '1.4', '0.0'  # Drone coordinates
        ],
    )

    return LaunchDescription([
        parseYaml,
        gazeboPx4,
        tmuxLauncher,
        tmuxAttach,
    ])