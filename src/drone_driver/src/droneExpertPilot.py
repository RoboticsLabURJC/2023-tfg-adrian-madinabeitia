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
import numpy as np

import ament_index_python
package_path = ament_index_python.get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.control_functions import PID, band_midpoint, search_top_line, search_bottom_line, save_timestamps, save_profiling, search_farthest_column

MIN_PIXEL = -360
MAX_PIXEL = 360

# Image parameters
BOTTOM_LIMIT_UMBRAL = 40
UPPER_LIMIT_UMBRAL = 10
UPPER_PROPORTION = 0.6
LOWER_PROPORTION = 0.4
BREAK_INCREMENT = 0.6

# Vel control
MAX_ANGULAR = 3.0
MAX_LINEAR = 6.0
MIN_LINEAR = 2.5
MAX_Z = 2.0

## PID controlers
ANG_KP = 2.0
ANG_KD = 1.75
ANG_KI = 0.0

Z_KP = 0.5
Z_KD = 0.45
Z_KI = 0.0

LIN_KP = 1.0
LIN_KD = 1.20
LIN_KI = 0.0


class droneController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = True) -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)

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

        # Create timers
        # self.timer = self.create_timer(0.1, self.timer_callback)
        self.saver = self.create_timer(5.0, self.save_data)


        #self.timer.

        # Frequency analysis 
        self.image_timestamps = []
        self.vel_timestamps = []
        self.profiling = []

        self.cv_image = None


    def save_data(self):
        self.get_logger().info("Saving data...")
        save_timestamps('./sub_timestamps.npy', self.image_timestamps)
        save_timestamps('./vel_timestamps.npy', self.vel_timestamps)
        save_profiling('./profiling_data.txt', self.profiling)

    def listener_callback(self, msg):
        self.image_timestamps.append(time.time())

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
        # Gets the drone velocitys
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
        # Gets the drone velocitys
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        # Sends the velocity command
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
    
    def get_angular_vel(self, farestPoint, nearestPoint):      
        widthCenter = self.cv_image.shape[1] / 2

        # Gets the angular error                    
        angularError = (widthCenter - farestPoint[0])*UPPER_PROPORTION + (widthCenter - nearestPoint[0])*LOWER_PROPORTION

        # Pixel distance to angular vel transformation
        angular = (((angularError - MIN_PIXEL) * self.ang_rang) / self.px_rang) + (-MAX_ANGULAR)
        anguarVel = self.angular_pid.get_pid(angular)

        return anguarVel
    
    def get_linear_vel(self, farestPoint):
        widthCenter = self.cv_image.shape[1] / 2

        pixelError = max(widthCenter, farestPoint) - min(widthCenter, farestPoint)

        error = np.interp(abs(pixelError), (0, widthCenter), (0, MAX_LINEAR-MIN_LINEAR))
        linearError = MAX_LINEAR - error * BREAK_INCREMENT
        linearVel = self.linear_pid.get_pid(linearError)

        return linearVel 

    def follow_line(self):
        if self.info['state'] == PlatformStatus.FLYING and self.cv_image is not None:
            initTime = time.time()

            # Gets the reference points
            ## Farest point
            topPoint = search_top_line(self.cv_image)
            farestPoint = band_midpoint(self.cv_image, topPoint, topPoint + UPPER_LIMIT_UMBRAL)

            # Nearest point
            bottomPoint = search_bottom_line(self.cv_image)
            nearestPoint = band_midpoint(self.cv_image, bottomPoint-BOTTOM_LIMIT_UMBRAL, bottomPoint)

            # Distance point
            distancePoint = search_farthest_column(self.cv_image)

            # Gets drone velocitys
            angularVel = self.get_angular_vel(farestPoint, nearestPoint)
            linearVel = self.get_linear_vel(distancePoint)
            #self.get_logger().info("Angular vel = %f || Linear vel = %f" % (angularVel, linearVel))

            # Set the velocity
            # self.set_vel(self.linearVel, 0, 0, anguarVel)
            # self.get_logger().info("Linear = %f  | Angular = %f" % (linearVel, angularVel))
            self.set_vel2D(linearVel, 0, MAX_Z, angularVel)

            self.vel_timestamps.append(time.time())
            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
    

def main(args=None):
    rclpy.init(args=args)

    # Controller node
    drone = droneController(drone_id="drone0", verbose=False, use_sim_time=True)
    
    # Takes off
    drone.take_off_process()

    # Start the flight
    while rclpy.ok():
        drone.follow_line()

        # Process a single iteration of the ROS event loop
        rclpy.spin_once(drone, timeout_sec=0.1)

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
