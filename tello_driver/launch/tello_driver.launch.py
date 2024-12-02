import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

namespace= "drone0"
sim_time = "true"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}

def exit_process_function(_launch_context, route, ns):
    subprocess.run(["bash", route, ns], check=True)

def generate_launch_description():
    utils_path = os.path.join(get_package_share_directory('tello_driver'), 'utils')
    sim_config = os.path.join(get_package_share_directory('tello_driver'), 'config')


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
        #cmd=['tmux', 'attach-session', '-t', namespace],
        name="attach",
    )



    return LaunchDescription([
        tmuxLauncher,
        tmuxAttach,
    ])