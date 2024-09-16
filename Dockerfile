FROM osrf/ros:humble-desktop

# Install Gazebo 11
RUN apt-get update && apt-get install -q -y \
    ros-${ROS_DISTRO}-gazebo-ros-pkgs \
    ros-${ROS_DISTRO}-ros-gz \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libimage-exiftool-perl \
  && apt-get -y autoremove \
  && apt-get clean autoclean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
  && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install evdev

WORKDIR /opt
RUN git clone https://github.com/naoki-mizuno/ds4drv --branch devel \
    && cd ds4drv \
    && python3 setup.py install

RUN mkdir -p /ws/src

WORKDIR /ws
RUN cd src \
    && git clone https://github.com/naoki-mizuno/ds4_driver \
        --branch ${ROS_DISTRO}-devel

RUN git clone https://github.com/aerostack2/aerostack2.git -b 1.0.9 src/aerostack2
RUN git clone https://github.com/pariaspe/2023-tfg-adrian-madinabeitia.git src/2023-tfg-adrian-madinabeitia
RUN apt update && rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build"

WORKDIR /
RUN git clone https://github.com/Adrimapo/project_crazyflie_gates.git

COPY ./entrypoint.bash /
ENTRYPOINT [ "/entrypoint.bash" ]
CMD [ "/bin/bash" ]