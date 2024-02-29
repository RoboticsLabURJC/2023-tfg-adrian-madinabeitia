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

# Package includes
package_path = ament_index_python.get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.control_functions import PID, save_timestamps, save_profiling
from include.models import pilotNet
from train import load_checkpoint
from include.data import dataset_transforms

# Vel control
MAX_Z = 2

## PID controlers
Z_KP = 0.5
Z_KD = 0.45
Z_KI = 0.0

package_path = ament_index_python.get_package_share_directory("drone_driver")


class droneNeuralController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = True, output_directory: str=".", network_directory: str=".") -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)

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

        timerPeriod = 0.01  # seconds
        saveDataPeriod = 5.0

        # Frequency analysis 
        self.image_timestamps = []
        self.vel_timestamps = []
        self.profiling = []

        self.cv_image = None
        self.imageTensor = None


        # Folder name
        self.profilingDir = output_directory
        if not os.path.exists(self.profilingDir):
            os.makedirs(self.profilingDir)


    def save_data(self):

        # Creats the directory 
        os.makedirs(self.profilingDir, exist_ok=True)

        # Saves all the data
        save_timestamps(self.profilingDir + '/sub_timestamps.npy', self.image_timestamps)
        save_timestamps(self.profilingDir + '/vel_timestamps.npy', self.vel_timestamps)
        save_profiling(self.profilingDir + '/profiling_data.txt', self.profiling)

    def listener_callback(self, msg):
        self.image_timestamps.append(time.time())

        # Converts the image to cv2 format
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image
        resized_image = cv2.resize(cv_image, (64, 64))
       
        # Convert the resized image to a tensor for inference
        initTime = time.time()
        self.image_array = np.array(resized_image)
        img_tensor = dataset_transforms(self.image_array).to(self.device)
        self.imageTensor = img_tensor.unsqueeze(0)

        self.profiling.append(f"Tensor conversion time = {time.time() - initTime}")

    # Retrys aerostack command if it failed
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
        # Gets the drone velocitys
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])
        
        # Z pid
        errorZ = float(pz) - self.position[2]
        vz = self.z_pid.get_pid(errorZ)

        # Sends the velocity command
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        

    def set_vel(self, vx, vy, vz, yaw):
        # Gets the drone velocitys
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
            vels = self.model(self.imageTensor)[0]
            vels[1] = vels[1] / 10

        self.profiling.append(f"\nAngular inference = {time.time() - initTime}")
        return vels


    def follow_line(self):
        if self.info['state'] == PlatformStatus.FLYING:
            initTime = time.time()

            # Gets drone velocitys
            vels = self.get_vels()
            self.get_logger().info("Linear inference = %f  | Angular inference = %f" % (vels[0], vels[1]))

            # Set the velocity
            self.set_vel2D(vels[0], 0, MAX_Z, vels[1])

            # Profiling
            self.vel_timestamps.append(time.time())
            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
    

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



