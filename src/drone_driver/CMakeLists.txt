cmake_minimum_required(VERSION 3.8)
project(drone_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tello_msgs REQUIRED)

# Local includes
include_directories(
 # include
  ${gazebo_ros_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${tello_msgs_INCLUDE_DIRS}
)


#=============
# Install
#=============

## Install Python scripts
install(
  PROGRAMS src/carControl.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/parseWorld.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/drone_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

## Install world and launch files
install(
  DIRECTORY launch worlds config models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
