FROM osrf/ros:humble-desktop

RUN sudo apt update && apt install -y \
    vim \
    python3-pip \
    libgazebo11 \
    python3-wheel \  
    ros-humble-geographic-msgs \
    ros-humble-xacro \
    gnome-terminal \
    tmux \
    tmuxinator \
    ros-humble-gazebo-dev \ 
    ros-humble-gazebo-ros\
    ros-humble-ament-cmake-clang-format \
    build-essential cmake git libeigen3-dev libyaml-cpp-dev libopencv-dev \
    python3-pip python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Install setuptools, rosdep, and colcon
RUN sudo pip install setuptools==58.2 \
    && pip install rosdep  future \
    && pip install colcon-common-extensions \
    && pip install torch matplotlib albumentations torchvision Pillow numpy


RUN sudo pip3 install kconfiglib jsonschema

# Install Gazebo
RUN curl -sSL http://get.gazebosim.org | sh

ENTRYPOINT ["/ros_entrypoint.sh"]

RUN mkdir -p root/workspace/src/px4
WORKDIR /root/workspace

# Clone the required repositorys 
RUN git clone https://github.com/RoboticsLabURJC/2023-tfg-adrian-madinabeitia.git src/tfg

#* Aerostack 2
RUN git clone https://github.com/aerostack2/aerostack2.git src/aerostack2

#* Other dependencies
RUN git clone https://github.com/JdeRobot/RoboticsInfrastructure.git --branch humble-devel src/RoboticsInfrastructure

#* px4
RUN git clone https://github.com/aerostack2/as2_platform_pixhawk.git -- src/px4/as2_platform_pixhawk
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git --branch v2.4.1 src/px4/Micro-XRCE-DDS-Agent
RUN git clone https://github.com/aerostack2/project_px4_vision src/px4/project_px4_vision
RUN git clone https://github.com/PX4/PX4-Autopilot.git --branch v1.14.0 src/px4/PX4-Autopilot
RUN git clone https://github.com/PX4/px4_msgs.git src/px4/px4_msgs

RUN git clone https://github.com/naoki-mizuno/ds4_driver.git src/ds4_driver

# Bash configuration
RUN echo source /usr/share/gazebo-11/setup.bash >> /root/.bashrc
RUN echo source /opt/ros/humble/setup.bash >> /root/.bashrc
RUN echo export GAZEBO_HOME=/usr/share/gazebo >> /root/.bashrc
RUN echo export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/usr/share/gazebo-11/models >> /root/.bashrc
RUN echo export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins >> /root/.bashrc
RUN echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins >> /root/.bashrc
RUN echo source /root/workspace/install/setup.bash >> /root/.bashrc

RUN echo export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/src/aerostack2/as2_simulation_assets/as2_ign_gazebo_assets/models >> /root/.bashrc
RUN echo export PX4_FOLDER=/src/px4/PX4-Autopilot >> /root/.bashrc
RUN echo export AEROSTACK2_PATH=/src/aerostack2 >> /root/.bashrc
RUN echo source $AEROSTACK2_PATH/as2_cli/setup_env.bash >> /root/.bashrc
RUN echo AEROSTACK2_SIMULATION_DRONE_ID=drone0 >> /root/.bashrc
RUN echo export AS2_GZ_ASSETS_SCRIPT_PATH=/root/workspace/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts >> /root/.bashrc


# Changes to specific commits some repositorys 
WORKDIR /root/workspace/src/px4/px4_msgs
RUN git checkout 7203046

WORKDIR /root/workspace/src/px4/as2_platform_pixhawk
RUN git checkout 7c55374

WORKDIR /root/workspace/src/aerostack2
RUN git checkout 96fdc4b

#* Rosdep dependencies 
WORKDIR /root/workspace
RUN rosdep update

WORKDIR /root/workspace/src/px4/PX4-Autopilot
RUN git submodule update --init --recursive

RUN chmod +x /root/workspace/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts/default_run.sh
RUN chmod +x /root/workspace/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts/run_sitl.sh
RUN chmod +x /root/workspace/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts/parse_json.py
RUN chmod +x /root/workspace/install/as2_gazebo_classic_assets/share/as2_gazebo_classic_assets/scripts/*.sh


RUN rosdep install --from-paths src --ignore-src -r -y

#RUN colcon build 


CMD ["bash"]
