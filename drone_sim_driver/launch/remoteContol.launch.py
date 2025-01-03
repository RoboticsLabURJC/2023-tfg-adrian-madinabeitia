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

    # Declare launch arguments
    out_dir = DeclareLaunchArgument(
        'output_dir',
        default_value="./outDir",
        description="Directory to save profiling files"
    )

    altitude_control = DeclareLaunchArgument(
        'altitude_control',
        default_value="false",
        description="If true, a neural network will be required for altitude control"
    )

    net_dir = DeclareLaunchArgument(
        'network_dir',
        default_value="__none__",
        description="Direction neural network"
    )

    dp_dir = DeclareLaunchArgument(
        'dp_dir',
        default_value="__none__",
        description="Altitude neural network"
    )

    control = Node(
        package='drone_sim_driver',
        executable='remoteControl.py',
        output='screen',
        arguments=[
            '--output_dir', LaunchConfiguration('output_dir'),
            '--altitude_control', LaunchConfiguration('altitude_control'),
            '--network_dir', LaunchConfiguration('network_dir'),
            '--dp_dir', LaunchConfiguration('dp_dir')
        ]
    )

    return LaunchDescription([
        out_dir,
        altitude_control,
        net_dir,
        dp_dir,
        controller,
        control,
    ])
