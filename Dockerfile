FROM osrf/ros:humble-desktop

# install initial docker dependencies
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
    ca-certificates \
		gnupg \
		lsb-core \
		sudo \
		wget \
  && rm -rf /var/lib/apt/lists/* 

RUN sudo apt update && apt install -y \
    libgazebo11 \
    libgazebo-dev \
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
    git-lfs \
&& rm -rf /var/lib/apt/lists/*

# Install Gazebo 11
RUN sudo rosdep fix-permissions \
  && rosdep update \
  && apt-get update && apt-get install -q -y \
    ros-humble-gazebo* \
    ros-humble-ros-gz-* \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    libimage-exiftool-perl \
  && apt-get -y autoremove \
  && apt-get clean autoclean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# PX4 general dependencies
RUN apt-get update && apt-get -y --quiet --no-install-recommends install \
    astyle \
    build-essential \
    cmake \
    cppcheck \
    file \
    g++ \
    gcc \
    gdb \
    git \
    lcov \
    libfuse2 \
    libxml2-dev \
    libxml2-utils \
    make \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    rsync \
    shellcheck \
    unzip \
    zip \
  && apt-get -y autoremove \
  && apt-get clean autoclean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*
  
# Install Python 3 pip build dependencies first
RUN python3.10 -m pip install --upgrade pip==23.3.1 wheel==0.41.3 setuptools==69.0.2
RUN python3.10 -m pip install evdev

# Install setuptools, rosdep, and colcon
RUN python3.10 -m pip install \
    # setuptools==58.2 \
    rosdep \
    future \
    colcon-common-extensions \
    torch==2.4.1 matplotlib albumentations torchvision Pillow numpy
RUN sudo pip3 install jsonschema==4.18.0

# Installing PX4 Python3 dependencies
RUN python3.10 -m pip install argparse==1.4.0 argcomplete==3.1.2 coverage==7.3.2 cerberus==1.3.5 \
    empy==3.3.4 jinja2==3.1.2 kconfiglib==14.1.0 matplotlib>=3.0 numpy==1.23.4 nunavut==1.1.0 \
    packaging==23.2 pkgconfig==1.5.5 pyros-genmsg==0.5.8 pyulog==1.0.1 pyyaml==6.0.1 \
    requests==2.22.0 serial==0.0.97 six==1.14.0 toml==0.10.2 sympy>=1.10.1 \
    psutil==5.9.0 utm==0.7.0 psycopg2 rosbags==0.10.6 tensorboard==2.18.0 ds4drv==0.5.1\

# Install PX4
RUN git clone -b v1.14.3 https://github.com/PX4/PX4-Autopilot.git --recursive \
  && cd /PX4-Autopilot \
  && DONT_RUN=1 make px4_sitl gazebo

RUN mkdir -p root/ws/src/px4
WORKDIR /root/ws

RUN git clone https://github.com/JdeRobot/RoboticsInfrastructure.git --branch humble-devel src/RoboticsInfrastructure
RUN git clone https://github.com/naoki-mizuno/ds4_driver.git src/ds4_driver
RUN git clone https://github.com/aerostack2/as2_platform_pixhawk.git -b 1.0.9 src/px4/as2_platform_pixhawk
RUN git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git --branch v2.4.1 src/px4/Micro-XRCE-DDS-Agent

##
RUN git clone https://github.com/PX4/px4_msgs.git src/px4/px4_msgs -b release/1.14

RUN git clone https://github.com/aerostack2/aerostack2.git -b 1.0.9 src/aerostack2
RUN git clone https://github.com/RoboticsLabURJC/2023-tfg-adrian-madinabeitia.git src/2023-tfg-adrian-madinabeitia
RUN git clone https://huggingface.co/datasets/Adrimapo/dataset_tfg_drone_simulation /root/ws/src/2023-tfg-adrian-madinabeitia/original_dataset
RUN apt update && rosdep update && rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install"

WORKDIR /
# used?
RUN git clone https://github.com/Adrimapo/project_crazyflie_gates.git
RUN git clone https://github.com/aerostack2/project_px4_vision src/px4/project_px4_vision

RUN pip3 install PySimpleGUI-4-foss
RUN echo "set -g mouse on" > ~/.tmux.conf 

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /root/ws/install/setup.bash" >> ~/.bashrc

# Additional library's 
RUN apt update && apt install -y dbus-x11 libcanberra-gtk-module libcanberra-gtk3-module
RUN apt update && apt install -y alsa-utils pulseaudio

WORKDIR /root/ws

COPY ./entrypoint.bash /
ENTRYPOINT [ "/entrypoint.bash" ]
CMD [ "/bin/bash" ]
