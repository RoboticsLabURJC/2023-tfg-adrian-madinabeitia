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
            'simulation_config_file': sim_config + '/world.json'
        }.items(),
    )

    drone_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', LaunchConfiguration('namespace'), 
            '-file', sdf_path,
            '-x', '0',
            '-y', '0',
            '-z', '1.4',
            '-Y', '0'
        ],
    )

    micro_xrce_cmd = ['MicroXRCEAgent', 'udp4', '-p', '8888']

    micro_xrce = ExecuteProcess(
    cmd = micro_xrce_cmd,
    output = 'screen'
    )

    # Aerostack2:

    aerial_platform = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('as2_platform_pixhawk'), 'launch'),
            '/pixhawk_launch.py']),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
            'simulation_config_file':  sim_config + '/world.json',
            'platform_config_file': sim_config + '/platform_config.yaml'
        }.items(),
    )
    state_estimator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('as2_state_estimator'), 'launch'),
            '/state_estimator_launch.py']),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
            'plugin_name': 'raw_odometry',
            'plugin_config_file': sim_config + '/state_estimator_odom.yaml'
        }.items(),
    )
    motion_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('as2_motion_controller'), 'launch'),
            '/controller_launch.py']),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
            'motion_controller_config_file': sim_config + '/motion_controller.yaml',
            'plugin_name': 'pid_speed_controller',
            'plugin_config_file': sim_config + '/pid_speed_controller.yaml'
        }.items(),
    )
    behaviors = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('as2_behaviors_motion'), 'launch'),
            '/motion_behaviors_launch.py']),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'use_sim_time': 'false',
            'takeoff_plugin_name': 'takeoff_plugin_position',
            'go_to_plugin_name': 'go_to_plugin_position',
            'follow_path_plugin_name': 'follow_path_plugin_position',
            'land_plugin_name': 'land_plugin_speed'
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='drone0',
                             description='Drone namespace.'),

        DeclareLaunchArgument('sim_time', default_value='true',
                             description='Use sim time bool'),
        gazebo,
        drone_spawner,
        
        # Aerostack2:
        #micro_xrce,
        aerial_platform,
        state_estimator,
        motion_controller,
        behaviors,
    ])