#!/bin/python3

import time
import math
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import sys
import numpy as np
import os

import argparse

import ament_index_python
package_path = ament_index_python.get_package_share_directory("drone_sim_driver")
sys.path.append(package_path)

from include.control_functions import PID, band_midpoint, search_top_line, search_bottom_line, save_timestamps, save_profiling, search_farthest_column

MIN_PIXEL = -360
MAX_PIXEL = 360

# Image parameters
BOTTOM_LIMIT_UMBRAL = 40
UPPER_LIMIT_UMBRAL = 10
UPPER_PROPORTION = 0.4
LOWER_PROPORTION = 0.6
BREAK_INCREMENT = 1.2

# Vel control
MAX_ANGULAR = 2.0
MAX_LINEAR = 6.0
MIN_LINEAR = 3.0
MAX_Z = 2.0

## PID controllers
ANG_KP = 1.5
ANG_KD = 1.4
ANG_KI = 0.0

Z_KP = 0.5
Z_KD = 0.45
Z_KI = 0.0

LIN_KP = 1.0
LIN_KD = 30.00
LIN_KI = 0.0

LINEAL_ARRAY_LENGTH = 150

# Frequency's 
VEL_PUBLISH_FREQ = 0.05
SAVE_FREQ = 5

# Image
NUM_POINTS = 10
SPACE = 20


class droneController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = True, profiling_directory: str = ".") -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)

        self.velPublisher_ = self.create_publisher(Point, 'commanded_vels', 10)
        self.imageSubscription = self.create_subscription(Image, '/filtered_img', self.listener_callback, 1)

        # PIDs
        self.angular_pid = PID(-MAX_ANGULAR, MAX_ANGULAR)
        self.angular_pid.set_pid(ANG_KP, ANG_KD, ANG_KI)

        self.z_pid = PID(-MAX_Z , MAX_Z)
        self.z_pid.set_pid(Z_KP, Z_KD, Z_KI)

        self.linear_pid = PID(MIN_LINEAR, MAX_LINEAR)
        self.linear_pid.set_pid(LIN_KP, LIN_KD, LIN_KI)
        
        # Control
        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.ang_rang = MAX_ANGULAR - (- MAX_ANGULAR)
        self.linearVel = MAX_LINEAR

        # Frequency analysis 
        self.generalTimestamps = []
        self.vel_timestamps = []
        self.profiling = []
        self.lastVels = []

        self.cv_image = None
        
        # Create the profiling directory if it doesn't exist
        self.profilingDir = profiling_directory
        if not os.path.exists(self.profilingDir):
            os.makedirs(self.profilingDir)


    def save_data(self):
        save_timestamps(self.profilingDir + '/general_timestamps.npy', self.generalTimestamps)
        save_timestamps(self.profilingDir + '/vel_timestamps.npy', self.vel_timestamps)
        save_profiling(self.profilingDir + '/profiling_data.txt', self.profiling)

    def listener_callback(self, msg):

        # Image conversion to cv2 format
        initTime = time.time()
        bridge = CvBridge()
        self.cv_image = bridge.imgmsg_to_cv2(msg, "mono8") 
        self.profiling.append(f"\nImage conversion time = {time.time() - initTime}")
        

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
    

    def take_off_process(self):
        self.get_logger().info("Start mission")

        ##### ARM OFFBOARD #####
        self.get_logger().info('Offboard')
        self.retry_command(self.offboard, lambda: self.info['offboard'])

        self.get_logger().info('Arm')
        self.retry_command(self.arm, lambda: self.info['armed'])


        ##### TAKE OFF #####
        self.get_logger().info("Take Off")
        self.takeoff(2.0, speed=1.0)
        while not self.info['state'] == PlatformStatus.FLYING:
            time.sleep(0.5)

        self.get_logger().info("Following line")

    
    def land_process(self, speed):
        self.get_logger().info("Landing")

        # Land process
        self.land(speed=speed)
        self.get_logger().info("Land done")

        # Disarms the drone
        self.disarm()
    

    def set_vel2D(self, vx, vy, pz, yaw):
        # Gets the drone velocity's
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        # Z pid
        errorZ = float(pz) - self.position[2]
        vz = self.z_pid.get_pid(errorZ)

        # Sends the velocity command
        initTime = time.time()
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        self.profiling.append(f"\nMotion_ref_handler = {time.time() - initTime}")
        

    def set_vel(self, vx, vy, vz, yaw):
        # Gets the drone velocity's
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        # Sends the velocity command
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
    
    def get_angular_vel(self, farthestPoint, nearestPoint):  
        widthCenter = self.cv_image.shape[1] / 2

        topPoint = search_top_line(self.cv_image)
        bottomPoint = search_bottom_line(self.cv_image)
        lineDiv = NUM_POINTS

        angularError = 0
        lastAng = 0
        intervals = (bottomPoint - topPoint) / lineDiv
        if int(intervals) != 0:
            
            angularError = 0
            for i in range(0, (bottomPoint - topPoint), int(intervals)):
                angularError += widthCenter - (band_midpoint(self.cv_image, topPoint + i - SPACE, topPoint + i + 1))[0]
            lastAng = angularError
            angularError = angularError /  lineDiv

        if intervals == 0 or abs(angularError == widthCenter):
            sign = 1
            if angularError != 0:
                sign = abs(lastAng) / lastAng

            angularError = widthCenter * sign

        # Gets the angular error                    
        # angularError = (widthCenter - farthestPoint[0])*UPPER_PROPORTION + (widthCenter - nearestPoint[0])*LOWER_PROPORTION

        # Pixel distance to angular vel transformation
        angular = (((angularError + widthCenter) * self.ang_rang) / self.cv_image.shape[1]) + (-MAX_ANGULAR)
        angularVel = self.angular_pid.get_pid(angular)

        return angularVel
    
    def get_linear_vel(self, farthestPoint):
        widthCenter = self.cv_image.shape[1] / 2

        pixelError = max(widthCenter, farthestPoint) - min(widthCenter, farthestPoint)
        error = np.interp(abs(pixelError), (0, widthCenter), (0, MAX_LINEAR-MIN_LINEAR))

        linearError = MAX_LINEAR - error * BREAK_INCREMENT

        linearVel = self.linear_pid.get_pid(linearError)

        if linearVel < MIN_LINEAR:
            linearVel = MIN_LINEAR

        return linearVel 

    def follow_line(self):
        vels = Point()
        if self.info['state'] == PlatformStatus.FLYING and self.cv_image is not None:
            initTime = time.time()

            # Gets the reference points
            ## Farthest point
            topPoint = search_top_line(self.cv_image)
            farthestPoint = band_midpoint(self.cv_image, topPoint, topPoint + UPPER_LIMIT_UMBRAL)

            # Nearest point
            bottomPoint = search_bottom_line(self.cv_image)
            nearestPoint = band_midpoint(self.cv_image, bottomPoint-BOTTOM_LIMIT_UMBRAL, bottomPoint)

            # Distance point
            distancePoint = search_farthest_column(self.cv_image)

            # Gets drone velocity's
            angularVel = self.get_angular_vel(farthestPoint, nearestPoint)
            linearVelRaw = self.get_linear_vel(distancePoint)

            # Smooths linear vel with a low pass filter
            self.lastVels.append(linearVelRaw)
            if len(self.lastVels) > LINEAL_ARRAY_LENGTH:
                self.lastVels.pop(0)
            
            linearVel = np.mean(self.lastVels)
            
            # Publish the info for training
            vels.x = float(linearVel)
            vels.y = float(linearVelRaw)
            vels.z = float(angularVel)
            self.velPublisher_.publish(vels)

            # Set the velocity
            self.get_logger().info("Linear = %f  | Angular = %f" % (linearVel, angularVel))
            self.set_vel2D(linearVel, 0, MAX_Z, angularVel)

            self.vel_timestamps.append(time.time())
            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
    

def main(args=None):
    rclpy.init(args=args)

    # Gets the necessary arguments
    parser = argparse.ArgumentParser(description='Drone Controller with Profiling', allow_abbrev=False)
    parser.add_argument('--output_directory', type=str, help='Directory to save profiling files', required=True)
    parsed_args, _ = parser.parse_known_args()

    # Controller node
    drone = droneController(drone_id="drone0", verbose=False, use_sim_time=True, profiling_directory=parsed_args.output_directory)
    
    # Takes off
    drone.take_off_process()
    
    initTime = time.time()
    # Start the flight
    while rclpy.ok():
        drone.follow_line()

        if time.time() - initTime >= SAVE_FREQ: 
            drone.save_data()
            initTime = time.time()

        drone.generalTimestamps.append(time.time())

        # Process a single iteration of the ROS event loop
        rclpy.spin_once(drone, timeout_sec=VEL_PUBLISH_FREQ)

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