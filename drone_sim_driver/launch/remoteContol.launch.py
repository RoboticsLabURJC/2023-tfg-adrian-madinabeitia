from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

namespace = "drone0"
sim_time = "true"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}

def generate_launch_description():

    controller = Node(
        package='ds4_driver',
        executable='ds4_driver_node.py',
    )

    out_dir = DeclareLaunchArgument(
        'out_dir',
        default_value="./outDir"
    )

    net_dir = DeclareLaunchArgument(
        'net_dir',
        default_value="None"
    )

    dp_dir = DeclareLaunchArgument(
        'dp_dir',
        default_value="None"
    )

    control = Node(
        package='drone_sim_driver',
        executable='remoteControl.py',
        output='screen',
        arguments=[
            '--output_dir', LaunchConfiguration('out_dir'),
            '--network_dir', LaunchConfiguration('net_dir'),
            '--dp_dir', LaunchConfiguration('dp_dir')
        ]
    )

    return LaunchDescription([
        net_dir,
        out_dir,
        dp_dir,
        controller,
        control,
    ])
