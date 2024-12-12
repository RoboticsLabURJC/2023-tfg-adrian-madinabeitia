from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

namespace= "drone0"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}

def generate_launch_description():

    # Arguments
    out_dir = DeclareLaunchArgument(
        'out_dir',
        default_value="."
    )

    trace_arg = DeclareLaunchArgument(
        'trace',
        default_value="false",
        description="Enable trace"
    )

    network_path = DeclareLaunchArgument(
        'network_path',
        default_value="."
    )

    filterImage = Node(
        package='drone_sim_driver',
        executable='image_filter_node.py',
        arguments=[
            '--output_directory', LaunchConfiguration('out_dir'),
            '--trace', LaunchConfiguration('trace')
        ],
    )

    control = Node(
        package='drone_sim_driver',
        executable='droneNeuralPilot.py',
        output='screen',
        arguments=[
            '--output_directory', LaunchConfiguration('out_dir'),
            '--network_directory', LaunchConfiguration('network_path')
        ],
    )  

    return LaunchDescription([
        out_dir,
        trace_arg,
        network_path,
        filterImage,
        control,
    ])