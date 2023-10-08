import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    world_path = os.path.join(get_package_share_directory('drone_driver'), 'worlds', 'ocean_simulation.world')
    sdf_path = os.path.join(get_package_share_directory('custom_robots'), 'models', 'iris_dual_cam/iris_dual_cam.sdf')
    sim_config = os.path.join(get_package_share_directory('drone_driver'), 'config')

    # Simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),

        launch_arguments={
            'world': world_path,
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': LaunchConfiguration('sim_time'),
            #'simulation_config_file': sim_config + '/world.json'
        }.items(),
    )

    drone_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', LaunchConfiguration('namespace'),  # Add your desired namespace here
            '-file', sdf_path,
            '-x', '0',
            '-y', '0',
            '-z', '1.4',
            '-Y', '0'
        ],
    )



    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='car',
                             description='Car name.'),

        DeclareLaunchArgument('sim_time', default_value='true',
                             description='Use sim time bool'),
        gazebo,
        drone_spawner,
    
    ])