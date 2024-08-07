cmake_minimum_required(VERSION 3.8)
project(drone_behaviors)

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
  PROGRAMS src/dataset/droneExpertPilot.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/dataset/remoteControl.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/dataset/image_filter_gates.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/models/droneNeuralPilot.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/features/processImages.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/dataset/image_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/dataset/get_gates_vel.py
  DESTINATION lib/${PROJECT_NAME}
)

## Install world and launch files
install(
  DIRECTORY launch utils src
  DESTINATION share/${PROJECT_NAME}
)



ament_package()
