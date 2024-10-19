FROM osrf/ros:humble-desktop

# # Install Gazebo 11
# RUN apt-get update && apt-get install -q -y \
#     ros-${ROS_DISTRO}-gazebo-ros-pkgs \
#     ros-${ROS_DISTRO}-ros-gz \
#     gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-ugly \
#     gstreamer1.0-libav \
#     libgstreamer-plugins-base1.0-dev \
#     libimage-exiftool-perl \
#   && apt-get -y autoremove \
#   && apt-get clean autoclean \
#   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y \
    apt-utils \
    software-properties-common \
    git \
    tmux \
    tmuxinator \
    vim \
    python-is-python3 \
    python3-rosdep \
    python3-pip \
    python3-colcon-common-extensions \
    python3-wheel \
  && rm -rf /var/lib/apt/lists/* 

  RUN sudo apt update && apt install -y \
    libgazebo11 \
    ros-humble-geographic-msgs \
    ros-humble-xacro \
    gnome-terminal \
    ros-humble-gazebo-dev \ 
    ros-humble-gazebo-ros\
    ros-humble-ament-cmake-clang-format \
    build-essential \
    cmake \
    libeigen3-dev \
    libyaml-cpp-dev \
    libopencv-dev \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install evdev

# Install setuptools, rosdep, and colcon
RUN sudo pip install setuptools==58.2 \
    && pip install rosdep  future \
    && pip install colcon-common-extensions \
    && pip install torch matplotlib albumentations torchvision Pillow numpy
RUN sudo pip3 install kconfiglib jsonschema

# Install Gazebo
RUN curl -sSL http://get.gazebosim.org | sh

RUN mkdir -p root/ws/src/px4
WORKDIR /root/ws

RUN git clone https://github.com/JdeRobot/RoboticsInfrastructure.git --branch humble-devel src/RoboticsInfrastructure
RUN git clone https://github.com/naoki-mizuno/ds4_driver.git src/ds4_driver
RUN git clone https://github.com/aerostack2/as2_platform_pixhawk.git -b 1.0.9 src/px4/as2_platform_pixhawk
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git --branch v2.4.1 src/px4/Micro-XRCE-DDS-Agent

##
RUN git clone https://github.com/PX4/px4_msgs.git src/px4/px4_msgs -b release/1.14
RUN git clone https://github.com/PX4/PX4-Autopilot.git --branch v1.14.3 --recursive src/px4/PX4-Autopilot

RUN git clone https://github.com/aerostack2/aerostack2.git -b 1.0.9 src/aerostack2
RUN git clone https://github.com/pariaspe/2023-tfg-adrian-madinabeitia.git src/2023-tfg-adrian-madinabeitia
RUN apt update && rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build"

WORKDIR /
RUN git clone https://github.com/Adrimapo/project_crazyflie_gates.git
RUN git clone https://github.com/aerostack2/project_px4_vision src/px4/project_px4_vision

COPY ./entrypoint.bash /
ENTRYPOINT [ "/entrypoint.bash" ]
CMD [ "/bin/bash" ]