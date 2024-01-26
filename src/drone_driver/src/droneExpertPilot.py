#!/bin/python3

import time
import math
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ..include.control_functions import PID, band_midpoint, search_top_line, search_bottom_line, save_timestamps, save_profiling

MIN_PIXEL = -360
MAX_PIXEL = 360

# Image parameters
LIMIT_UMBRAL = 40
UPPER_PROPORTION = 0.6
LOWER_PROPORTION = 0.4

# Vel control
MAX_ANGULAR = 4
MAX_LINEAR = 2.5
MAX_Z = 2

## PID controlers
ANG_KP = 0.8
ANG_KD = 0.75
ANG_KI = 0.0

Z_KP = 0.5
Z_KD = 0.45
Z_KI = 0.0


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
        
        # Control
        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.ang_rang = MAX_ANGULAR - (- MAX_ANGULAR)
        self.linearVel = MAX_LINEAR

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer = self.create_timer(5.0, self.save_data)

        # Frequency analysis 
        self.image_timestamps = []
        self.vel_timestamps = []
        self.profiling = []

        self.cv_image = None


    def save_data(self):

        save_timestamps('./sub_timestamps.npy', self.image_timestamps)
        save_timestamps('./vel_timestamps.npy', self.vel_timestamps)
        save_profiling('./profiling_data.txt', self.profiling)

    def listener_callback(self, msg):
        self.image_timestamps.append(time.time())

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
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        errorZ = float(pz) - self.position[2]
        vz = self.z_pid.get_pid(errorZ)

        initTime = time.time()
        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
        self.profiling.append(f"\nMotion_ref_handler = {time.time() - initTime}")
        

    def set_vel(self, vx, vy, vz, yaw):
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed(
            [float(velX), float(velY), float(vz)], 'earth', float(yaw))
    
    def get_angular_vel(self):
        getAngTime = time.time()
        if self.cv_image is None:
            return 0.0
        
        width_center = self.cv_image.shape[1] / 2

        # Searchs limits of the line
        initTime = time.time()
        top_point = search_top_line(self.cv_image)
        bottom_point = search_bottom_line(self.cv_image)
        self.profiling.append(f"\nPoint getter= {time.time() - initTime}")

        # Searchs the reference points
        initTime = time.time()
        red_farest = band_midpoint(self.cv_image, top_point, top_point + LIMIT_UMBRAL)
        # red_nearest = band_midpoint(self.cv_image, bottom_point-LIMIT_UMBRAL, bottom_point)
        self.profiling.append(f"\nReference point getter= {time.time() - initTime}")
                    
        #angular_distance = (width_center - red_farest[0])*UPPER_PROPORTION + (width_center - red_nearest[0])*LOWER_PROPORTION
        angular_distance = (width_center - red_farest[0])


        # Pixel distance to angular vel transformation
        angular = (((angular_distance - MIN_PIXEL) * self.ang_rang) / self.px_rang) + (-MAX_ANGULAR)
        anguarVel = self.angular_pid.get_pid(angular)

        self.profiling.append(f"\nGet angular time= {time.time() - getAngTime}")
        return anguarVel

    def timer_callback(self):
        if self.info['state'] == PlatformStatus.FLYING:
            

            initTime = time.time()
            angular_vel = self.get_angular_vel()
            self.get_logger().info("Angular vel = %f" % angular_vel)


            # self.set_vel(self.linearVel, 0, 0, anguarVel)
            self.set_vel2D(self.linearVel, 0, 2.0, angular_vel)

            self.vel_timestamps.append(time.time())

            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
        return
    

def main(args=None):
    height = 2.0

    rclpy.init(args=args)

    drone = droneController(drone_id="drone0", verbose=False, use_sim_time=True)
    drone.take_off_process(height)

    time.sleep(1)
    rclpy.spin(drone)
        


    drone.destroy_node()

    try:
        rclpy.shutdown()
        print("Clean exit")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
    exit()


if __name__ == '__main__':
    main()





