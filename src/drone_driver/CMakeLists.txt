cmake_minimum_required(VERSION 3.8)
project(drone_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Find packages
find_package(ament_cmake REQUIRED)
#find_package(gazebo REQUIRED)  # Note uppercase variables in /usr/lib/x86_64-linux-gnu/cmake/gazebo/gazebo-config.cmake
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tello_msgs REQUIRED)

# Local includes
include_directories(
  include
  ${gazebo_ros_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${tello_msgs_INCLUDE_DIRS}
)

#=============
# Tello Gazebo plugin
#=============

add_library(
  TelloPlugin SHARED
  src/basic_plugin.cpp
)

ament_target_dependencies(
  TelloPlugin
  gazebo_dev
  gazebo_ros
  geometry_msgs
  rclcpp
  std_msgs
  tello_msgs
)

#=============
# Install
#=============

# Install targets
install(
  TARGETS TelloPlugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

## Install Python scripts
install(
  PROGRAMS src/inject_entity.py
  DESTINATION lib/${PROJECT_NAME}
)

## Install world and launch files
install(
  DIRECTORY launch worlds urdf config models
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
