#!/bin/python3

import time
import math
import argparse
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from as2_motion_reference_handlers.speed_motion import SpeedMotion
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float32

from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

from control_functions import PID, band_midpoint, search_top_line, search_bottom_line

MIN_PIXEL = -360
MAX_PIXEL = 360

# Image parameters
LIMIT_UMBRAL = 40
UPPER_PROPORTION = 0.7
LOWER_PROPORTION = 0.3

# Vel control
MAX_ANGULAR = 2
MAX_LINEAR = 2
MAX_Z = 2

## PID controlers
ANG_KP = 0.8
ANG_KD = 0.75
ANG_KI = 0.0

Z_KP = 0.8
Z_KD = 0.4
Z_KI = 0.0


class droneController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = False) -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)

        self.imageSubscription = self.create_subscription(Image, '/filtered_img', self.listener_callback, 1)

        # PIDs
        self.angular_pid = PID(-MAX_ANGULAR, MAX_ANGULAR)
        self.angular_pid.set_pid(ANG_KP, ANG_KD, ANG_KI)

        self.z_pid = PID(-MAX_Z , MAX_Z)
        self.z_pid.set_pid(Z_KP, Z_KD, Z_KI)
        
        # Control
        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.ang_rang = MAX_ANGULAR - (- MAX_ANGULAR)
        self.linearVel = MAX_LINEAR
        self.anguarVel = 0   # Updates in the listener callback



    def listener_callback(self, msg):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "mono8") 
        img_height = cv_image.shape[0]

        if True:
            width_center = cv_image.shape[1] / 2

            top_point = search_top_line(cv_image)
            bottom_point = search_bottom_line(cv_image)
            red_farest = band_midpoint(cv_image, top_point, top_point + LIMIT_UMBRAL)
            red_nearest = band_midpoint(cv_image, bottom_point-LIMIT_UMBRAL*2, bottom_point)
                
            angular_distance = (width_center - red_farest[0])#*UPPER_PROPORTION + (width_center - red_nearest[0])*LOWER_PROPORTION

            # Pixel distance to angular vel transformation
            angular = (((angular_distance - MIN_PIXEL) * self.ang_rang) / self.px_rang) + (-MAX_ANGULAR)
            self.anguarVel = self.angular_pid.get_pid(angular)
            # self.draw_traces(cv_image)


    def retry_command(self, command, check_func, sleep_time=1.0, max_retries=1):
        if not check_func():
            command()
            count = 0

            if check_func():
                return 
            
            while not check_func() or count < max_retries:
                print("Retrying...")
                time.sleep(sleep_time)
                command()
                count += 1

            if not check_func():
                print("Command failed")
    

    def take_off_process(self, takeoff_height):
        print("Start mission")

        ##### ARM OFFBOARD #####
        print('Offboard')
        self.retry_command(self.offboard, lambda: self.info['offboard'])
        print("Offboard done")

        print('Arm')
        self.retry_command(self.arm, lambda: self.info['armed'])
        print("Arm done")


        ##### TAKE OFF #####
        print("Take Off")
        self.takeoff(takeoff_height, speed=1.0)
        while not self.info['state'] == PlatformStatus.FLYING:
            time.sleep(0.5)

        print("Take Off done")

    
    def land_process(self, speed):
        print("Landing")

        self.land(speed=speed)
        print("Land done")

        self.disarm()
    

    def set_vel2D(self, vx, vy, pz, yaw):
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        errorZ = float(pz) - self.position[2]
        vz = self.z_pid.get_pid(errorZ)

        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        

    def set_vel(self, vx, vy, vz, yaw):
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        

    def velocityControl(self, height):
        if self.info['state'] == PlatformStatus.FLYING:
            # self.set_vel(self.linearVel, 0, 0, self.anguarVel)
            self.set_vel2D(self.linearVel, 0, height, self.anguarVel)
            

if __name__ == '__main__':

    height = 0.5

    rclpy.init()

    drone = droneController(drone_id="drone0", verbose=False, use_sim_time=True)
    drone.take_off_process(height)

    try:
        while True:
            drone.velocityControl(height)
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Keyboard interrupt.")

    drone.land()
    drone.destroy_node()

    try:
        rclpy.shutdown()
        print("Clean exit")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    exit()





