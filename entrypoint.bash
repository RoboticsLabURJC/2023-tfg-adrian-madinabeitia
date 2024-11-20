#!/bin/bash

if [[ -f /opt/ros/$ROS_DISTRO/setup.bash ]]; then
    source /opt/ros/$ROS_DISTRO/setup.bash
fi

if [[ -f /root/ws/install/setup.bash ]]; then
    source /root/ws/install/setup.bash
fi

source /usr/share/gazebo-11/setup.bash
export GAZEBO_HOME=/usr/share/gazebo
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/usr/share/gazebo-11/models
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/ws/src/aerostack2/as2_simulation_assets/as2_gazebo_assets/models
export PX4_FOLDER=/root/ws/src/px4/PX4-Autopilot
export AEROSTACK2_PATH=/root/ws/src/aerostack2
source $AEROSTACK2_PATH/as2_cli/setup_env.bash
AEROSTACK2_SIMULATION_DRONE_ID=drone0
export AS2_GZ_ASSETS_SCRIPT_PATH=/root/ws/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts

exec "$@"