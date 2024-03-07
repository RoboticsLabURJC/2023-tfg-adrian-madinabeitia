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

    filterImage = Node(
        package='drone_driver',
        executable='image_filter_node.py',
        arguments=[
            '--output_directory', LaunchConfiguration('out_dir'),
            '--trace', LaunchConfiguration('trace')
        ],
    )

    out_dir = DeclareLaunchArgument(
        'out_dir',
        default_value="."
    )

    trace_arg = DeclareLaunchArgument(
        'trace',
        default_value="false",
        description="Enable trace"
    )

    control = Node(
        package='drone_driver',
        executable='droneExpertPilot.py',
        output='screen',
        arguments=[
            '--output_directory', LaunchConfiguration('out_dir'),
        ]
    )

    return LaunchDescription([
        out_dir,
        trace_arg,
        filterImage,
        control,
    ])
