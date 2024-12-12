import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

namespace = "drone0"
sim_time = "true"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}


def exit_process_function(_launch_context, route, ns):
    subprocess.run(["bash", route, ns], check=True)


def generate_launch_description():
    sim_config = os.path.join(get_package_share_directory('drone_sim_driver'), 'config')
    utils_path = os.path.join(get_package_share_directory('drone_sim_driver'), 'utils')
    worlds_path = os.path.join(get_package_share_directory('drone_sim_driver'), 'worlds')

    world = DeclareLaunchArgument(
        'world',
        default_value="nurburgring_line.world"
    )
    yaw = DeclareLaunchArgument(
        'yaw',
        default_value="0.0"
    )

    # Px4 autopilot gazebo launch
    gazeboPx4 = ExecuteProcess(
        cmd=[
            '/bin/bash', '-c',
            f'$AS2_GZ_ASSETS_SCRIPT_PATH/default_run.sh {sim_config}/world.json'
        ],
        name="gazebo",
        additional_env=env_vars,

        # Closes the tmux session on gazebo exit
        on_exit=[
            OpaqueFunction(
                function=exit_process_function,
                args=[utils_path + '/end_tmux.sh', namespace]
            ),
            LogInfo(msg='Tmux session closed')
        ]
    )

    # Prepares the tmux session
    tmuxLauncher = ExecuteProcess(
        cmd=['tmuxinator', 'start', '-n', namespace, '-p', sim_config + '/tmuxLaunch.yml',
             "drone_namespace=" + namespace,
             "simulation_time=" + sim_time,
             "config_path=" + sim_config],

        name="tmuxLauncher",
        output='screen'
    )

    tmuxAttach = ExecuteProcess(
        # Aerostack2 terminal
        cmd=['gnome-terminal', '--', 'tmux', 'attach-session', '-t', namespace],

        # No additional window
        # cmd=['tmux', 'attach-session', '-t', namespace],
        name="attach",
    )

    parseYaml = Node(
        package='drone_sim_driver',
        executable='parseWorld.py',
        output='screen',
        arguments=[
            sim_config + '/world.json',
            [worlds_path, '/', LaunchConfiguration('world')],
            '0.0', '0.0', '0.0', LaunchConfiguration('yaw')  # 3.14 # 3.54
        ],
    )

    return LaunchDescription([
        world,
        yaw,
        parseYaml,
        gazeboPx4,
        tmuxLauncher,
        tmuxAttach,
    ])
