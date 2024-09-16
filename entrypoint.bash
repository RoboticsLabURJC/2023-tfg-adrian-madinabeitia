#!/bin/bash

if [[ -f /opt/ros/$ROS_DISTRO/setup.bash ]]; then
    source /opt/ros/$ROS_DISTRO/setup.bash
fi

if [[ -f /ws/install/setup.bash ]]; then
    source /ws/install/setup.bash
fi

exec "$@"