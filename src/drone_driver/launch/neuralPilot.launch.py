import os
from launch import LaunchDescription
from launch_ros.actions import Node

namespace= "drone0"
sim_time = "true"
world = "/simple_circuit.world"

env_vars = {
    'AEROSTACK2_SIMULATION_DRONE_ID': namespace
}

def generate_launch_description():

    filterImage = Node(
        package='drone_driver',
        executable='image_filter_node.py',
        # output='screen'
    )  

    control = Node(
        package='drone_driver',
        executable='droneNeuralPilot.py',
        output='screen'
    )  

    return LaunchDescription([
        filterImage,
        control,
    ])