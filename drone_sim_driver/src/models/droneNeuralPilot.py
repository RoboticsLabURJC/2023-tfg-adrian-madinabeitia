#!/bin/python3

import time
import math
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
import cv2
import numpy as np
import ament_index_python
import torch
import argparse
import os
from geometry_msgs.msg import Point

# Package includes
package_path = ament_index_python.get_package_share_directory("drone_sim_driver")
sys.path.append(package_path)

from src.control_functions import PID, save_timestamps, save_profiling
from src.models.models import pilotNet
from src.models.train import load_checkpoint
from src.dataset.data import dataset_transforms, MAX_ANGULAR, MIN_LINEAR, MAX_LINEAR

# Vel control
MAX_Z = 2
LINEAL_ARRAY_LENGTH = 150
ANGULAR_ARRAY_LENGTH = 1
REDUCTION = 3.5

## PID controllers
Z_KP = 0.5
Z_KD = 0.45
Z_KI = 0.0


class droneNeuralController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = True, output_directory: str=".", network_directory: str=".") -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)

        self.velPublisher_ = self.create_publisher(Point, 'commanded_vels', 10)
        self.imageSubscription = self.create_subscription(Image, '/drone0/sensor_measurements/frontal_camera/image_raw', self.listener_callback, 1)

        # Gets the trained model
        self.model = pilotNet()
        load_checkpoint(network_directory, self.model)
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        # PID
        self.z_pid = PID(-MAX_Z , MAX_Z)
        self.z_pid.set_pid(Z_KP, Z_KD, Z_KI)
        
        # Control
        self.linearVel = 0

        # Frequency analysis 
        self.generalTimestamps = []
        self.vel_timestamps = []
        self.profiling = []

        self.cv_image = None
        self.imageTensor = None
        self.lastVels = []
        self.lastAngular = []

        # Folder name
        self.profilingDir = output_directory
        if not os.path.exists(self.profilingDir):
            os.makedirs(self.profilingDir)


    def save_data(self):

        # Creates the directory 
        os.makedirs(self.profilingDir, exist_ok=True)

        # Saves all the data
        save_timestamps(self.profilingDir + '/general_timestamps.npy', self.generalTimestamps)
        save_timestamps(self.profilingDir + '/vel_timestamps.npy', self.vel_timestamps)
        save_profiling(self.profilingDir + '/profiling_data.txt', self.profiling)

    def listener_callback(self, msg):
        # Converts the image to cv2 format
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        resized_image = cv2.resize(cv_image, (64, 64))

        self.image_array = np.array(resized_image)
       
        # Convert the resized image to a tensor for inference
        initTime = time.time()
        img_tensor = dataset_transforms(self.image_array).to(self.device)
        self.imageTensor = img_tensor.unsqueeze(0)

        self.profiling.append(f"Tensor conversion time = {time.time() - initTime}")

    # Retry aerostack command if it failed
    def retry_command(self, command, check_func, sleep_time=1.0, max_retries=1):
        if not check_func():
            command()
            count = 0

            if check_func():
                return 
            
            while not check_func() or count < max_retries:
                self.get_logger().info("Retrying...")
                time.sleep(sleep_time)
                command()
                count += 1

            if not check_func():
                self.get_logger().info("Command failed")
    

    def take_off_process(self, takeoff_height):
        self.get_logger().info("Start mission")

        ##### ARM OFFBOARD #####
        self.get_logger().info('Offboard')
        self.retry_command(self.offboard, lambda: self.info['offboard'])

        self.get_logger().info('Arm')
        self.retry_command(self.arm, lambda: self.info['armed'])


        ##### TAKE OFF #####
        self.get_logger().info("Take Off")
        self.takeoff(takeoff_height, speed=1.0)
        while not self.info['state'] == PlatformStatus.FLYING:
            time.sleep(0.5)

        self.get_logger().info("Following line")

    
    def land_process(self, speed):
        self.get_logger().info("Landing")

        self.land(speed=speed)
        self.get_logger().info("Land done")

        self.disarm()
    

    def set_vel2D(self, vx, vy, pz, yaw):
        # Gets the drone velocity's
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])
        
        # Z pid
        errorZ = float(pz) - self.position[2]
        vz = self.z_pid.get_pid(errorZ)

        # Sends the velocity command
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        

    def set_vel(self, vx, vy, vz, yaw):
        # Gets the drone velocity's
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        # Sends the velocity command
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
    
    def get_vels(self):
        initTime = time.time()
        vels = (0, 0)

        # Angular inference for neural network
        if self.imageTensor is not None:
            vels = self.model(self.imageTensor)[0].tolist()

        self.profiling.append(f"\nAngular inference = {time.time() - initTime}")
        
        # Gets the vels
        angular = ((vels[1] * (2 * MAX_ANGULAR))  - MAX_ANGULAR) / REDUCTION
        lineal = ((vels[0] * (MAX_LINEAR - MIN_LINEAR)) - MIN_LINEAR) / 2

        return (lineal, angular)


    def follow_line(self):
        if self.info['state'] == PlatformStatus.FLYING:
            initTime = time.time()

            # Gets drone velocity's
            vels = self.get_vels()
            
            # Smooths linear vel with a low pass filter
            self.lastVels.append(vels[0])
            if len(self.lastVels) > LINEAL_ARRAY_LENGTH:
                self.lastVels.pop(0)
            
            # self.lastAngular.append(vels[1])
            # if len(self.lastAngular) > ANGULAR_ARRAY_LENGTH:
            #     self.lastAngular.pop(0)
            
            linearVel = np.mean(self.lastVels)
            #lastAngular = np.mean(self.lastAngular)
            lastAngular = vels[1]

            # Set the velocity
            self.set_vel2D(float(linearVel), 0, MAX_Z, float(lastAngular))

            # Profiling
            self.vel_timestamps.append(time.time())
            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
            
            # Publish the info for training
            velsPubMsg = Point()
            velsPubMsg.x = float(vels[0]) # Raw lineal vel
            velsPubMsg.y = float(linearVel) # Filtered lineal vel
            velsPubMsg.z = vels[1] # Angular vel
            self.velPublisher_.publish(velsPubMsg)

            # Logger
            self.get_logger().info("Linear inference = %f  | Angular inference = %f" % (vels[0], vels[1]))


def main(args=None):
    rclpy.init(args=args)
    
    # Gets the necessary arguments
    parser = argparse.ArgumentParser(description='Drone Controller with Profiling', allow_abbrev=False)
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--network_directory', type=str, required=True)
    parsed_args, _ = parser.parse_known_args()

    # Controller node
    drone = droneNeuralController(drone_id="drone0", verbose=False, use_sim_time=True, output_directory=parsed_args.output_directory, network_directory=parsed_args.network_directory)

    # Takes off
    drone.take_off_process(2.0)
    
    initTime = time.time()
    # Start the flight
    while rclpy.ok():
        drone.follow_line()

        if time.time() - initTime >= 5: 
            drone.save_data()
            initTime = time.time()

        drone.generalTimestamps.append(time.time())

        # Process a single iteration of the ROS event loop
        rclpy.spin_once(drone, timeout_sec=0.05)

    # End of execution
    drone.destroy_node()

    try:
        rclpy.shutdown()
        print("Clean exit")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    exit()

if __name__ == '__main__':
    main()


