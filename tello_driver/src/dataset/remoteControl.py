#!/bin/python3

import time
import math
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from ds4_driver_msgs.msg._status import Status
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.qos import QoSProfile, QoSPresetProfiles
import sys
import numpy as np
import signal
import os
from rclpy.serialization import serialize_message
from std_msgs.msg import String
import argparse
import rosbag2_py
import threading

from tf2_msgs.msg import TFMessage


import ament_index_python
package_path = ament_index_python.get_package_share_directory("tello_driver")
sys.path.append(package_path)

from src.control_functions import PID, save_timestamps, save_profiling


# Low pass filter
LINEAL_ARRAY_LENGTH = 150

# Frequency's 
VEL_PUBLISH_FREQ = 0.05
SAVE_FREQ = 5



class droneController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = True, profiling_directory: str = ".") -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)
        actualTime = time.time()

        # Topics names
        self.velTopic = "/drone0/commanded_vels"
        self.imageTopic = "/drone0/sensor_measurements/frontal_camera/image_raw"
        self.tfTopic = "/tf"

        # Velocity publisher
        self.velPublisher_ = self.create_publisher(Point, 'commanded_vels', 10)

        # Controller subscription
        self.controllerSubscription = self.create_subscription(Status, '/status', self.controller_callback, 1)
        self.imageSubscription = self.create_subscription(Image, self.imageTopic, self.image_callback, 1)
        self.imageSubscription = self.create_subscription(TFMessage, self.tfTopic, self.tf_callback, 1)
        
        # Controller joysticks
        self.leftX = 0
        self.leftY = 0
        self.rightX = 0
        self.rightY = 0

        # Limits
        self.max_angular = 3
        self.min_angular = 0.5
        self.max_linear = 12.0
        self.min_linear = 1
        self.max_z = 4.0
        self.min_z = 0.5

        # Button controllers
        self.buttonPeriod = 0.2
        self.joystickPeriod = 0.1
        self.lastL2 = actualTime
        self.lastR2 = actualTime
        self.lastL1 = actualTime
        self.lastR1 = actualTime
        self.lastCommanded = actualTime

        # Defaults
        self.angular_limit = self.max_angular / 2
        self.linear_limit = self.max_linear / 2
        self.posZ = 2.0
        
        # Frequency analysis 
        self.generalTimestamps = []
        self.vel_timestamps = []
        self.profiling = []
        self.lastVels = []
        
        # Create the profiling directory if it doesn't exist
        self.profilingDir = profiling_directory
        if not os.path.exists(self.profilingDir):
            os.makedirs(self.profilingDir)

        # Recorder
        self.firstRecord = True
        self.recordRosbag = False
        self.recordId = 0

        self.takeOff = False
        self.landBool = False


    def open_bag(self):
        ## Creates the writer
        self.writer = rosbag2_py.SequentialWriter()
        self.converter_options = rosbag2_py._storage.ConverterOptions('', '')

        storage_options = rosbag2_py._storage.StorageOptions(
            uri=self.profilingDir + '/rosbag',
            storage_id='sqlite3')
            
        self.writer.open(storage_options, self.converter_options)

        # Saves the desired topics
        ## Velocity topic
        vel_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.velTopic,
            type='geometry_msgs/msg/Point',
            serialization_format='cdr')

        ## Image topic
        image_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.imageTopic,
            type='sensor_msgs/msg/Image',
            serialization_format='cdr')

        ## Tf topic
        tf_topic_info = rosbag2_py._storage.TopicMetadata(
            name=self.tfTopic,
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr')

        self.writer.create_topic(vel_topic_info)
        self.writer.create_topic(image_topic_info)
        self.writer.create_topic(tf_topic_info) 

    def image_callback(self, msg):
        ## Saves the topic if desired
        if self.recordRosbag:
            self.writer.write(
                self.imageTopic,
                serialize_message(msg),
                self.get_clock().now().nanoseconds)        
    
    def tf_callback(self, msg):
        ## Saves the topic if desired
        if self.recordRosbag:
            self.writer.write(
                self.tfTopic,
                serialize_message(msg),
                self.get_clock().now().nanoseconds)
        

    def save_data(self):
        profilingDir = self.profilingDir + "/profiling"
        # Creates the directory
        os.makedirs(profilingDir, exist_ok=True)

        # Data for profiling
        save_timestamps(profilingDir + '/general_timestamps.npy', self.generalTimestamps)
        save_timestamps(profilingDir + '/vel_timestamps.npy', self.vel_timestamps)
        save_profiling(profilingDir + '/profiling_data.txt', self.profiling)

    def take_off_process(self):
        self.takeOff = False
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

        self.get_logger().info("Ready to operate")

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

    def set_vel2D(self, vx, vy, pz, yaw):
        # Gets the drone velocity's
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        # Z position control
        errorZ = float(pz) - self.position[2]
        vz = errorZ * 2.5

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
    

    def land_process(self):
        self.get_logger().info("Landing...")
        self.land(0.5)

        self.get_logger().info("Disarming drone...")
        self.retry_command(self.disarm, lambda: self.info['disarmed'])

    def cross_control(self, msg):
        
        # Up = Take off
        if msg.button_dpad_up == 1 and not self.takeOff:
            self.takeOff = True
        
        # Down = Land
        if msg.button_dpad_down == 1 and not self.landBool:
            self.landBool = True
    
    def velocity_control(self, msg):
        # Z axis position control
        if abs(msg.axis_right_y) > 0.5 and (time.time() - self.lastCommanded) > self.joystickPeriod:
            # Sets the correct sign
            self.posZ += 0.1 * abs(msg.axis_right_y) / msg.axis_right_y

            # Security limits
            if self.posZ > self.max_z:
                self.posZ = self.max_z
            
            if self.posZ < self.min_z:
                self.posZ = self.min_z

            self.lastCommanded = time.time()
        
        # Linear vel control
        if msg.button_l2 == 1 and (time.time() - self.lastL2) > self.buttonPeriod:
            self.linear_limit -= 0.5

            if self.linear_limit < self.min_linear:
                self.linear_limit = self.min_linear
            
            self.lastL2 = time.time()
            self.get_logger().info("Max linear vel = %f" % self.linear_limit)
        
        if msg.button_r2 == 1 and (time.time() - self.lastR2) > self.buttonPeriod:
            self.linear_limit += 0.5

            if self.linear_limit > self.max_linear:
                self.linear_limit = self.max_linear
            
            self.lastR2 = time.time()
            self.get_logger().info("Max linear vel = %f" % self.linear_limit)
        
        # Angular control
        if msg.button_l1 == 1 and (time.time() - self.lastL1) > self.buttonPeriod:
            self.angular_limit -= 0.5

            if self.angular_limit < self.min_angular:
                self.angular_limit = self.min_angular
            
            self.lastL1 = time.time()
            self.get_logger().info("Max angular vel = %f" % self.angular_limit)
        
        if msg.button_r1 == 1 and (time.time() - self.lastR1) > self.buttonPeriod:
            self.angular_limit += 0.5

            if self.angular_limit > self.max_angular:
                self.angular_limit = self.max_angular
            
            self.lastR1 = time.time()
            self.get_logger().info("Max angular vel = %f" % self.angular_limit)


    def controller_callback(self, msg):
        # Gets the joysticks data
        self.leftY = msg.axis_left_x
        self.leftX = msg.axis_left_y
        self.rightY = msg.axis_right_x
        self.rightX = msg.axis_right_y

        self.velocity_control(msg)

        # Take off/ Land control
        self.cross_control(msg)
        
        # Recording buttons
        if (msg.button_square == 1 and not self.recordRosbag):
            if self.firstRecord:
                self.open_bag()
                self.firstRecord = False

            self.get_logger().info("Recording...")
            self.recordRosbag = True
        
        if (msg.button_circle == 1 and self.recordRosbag):
            self.get_logger().info("Recording stooped...")
            self.recordRosbag = False


    def remote_control(self):
        vels = Point()
        if self.info['state'] == PlatformStatus.FLYING:
            initTime = time.time()

            # Gets drone velocity's
            angularVel =  self.leftY * self.angular_limit
            linearVelRaw = self.leftX * self.linear_limit
            lateralVel = self.rightY * self.linear_limit / 1.5

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

            if self.recordRosbag:
                self.writer.write(
                    self.velTopic,
                    serialize_message(vels),
                    self.get_clock().now().nanoseconds)  

            # Set the velocity
            # self.get_logger().info("Linear = %f  | Angular = %f" % (linearVel, angularVel))
            self.set_vel2D(linearVel, lateralVel, self.posZ, angularVel)

            self.vel_timestamps.append(time.time())
            self.profiling.append(f"\nTimer callback = {time.time() - initTime}")
    
def goal():
    time.sleep(5)
    print("Aaaaaaaa")

def sigint_handler(signum, frame, drone):
    # Lands the drone and clean all 
    drone.land_process()
    drone.destroy_node()

    try:
        rclpy.shutdown()
        print("Clean exit")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    print("Ctrl+C pressed. Exiting gracefully.")


def main(args=None):
    
    rclpy.init(args=args)

    # Gets the necessary arguments
    parser = argparse.ArgumentParser(description='Drone Controller with Profiling', allow_abbrev=False)
    parser.add_argument('--output_directory', type=str, help='Directory to save profiling files', required=True)
    parsed_args, _ = parser.parse_known_args()
    signal.signal(signal.SIGINT, lambda signum, frame: sigint_handler(signum, frame, drone))


    # Controller node
    drone = droneController(drone_id="drone0", verbose=False, use_sim_time=True, profiling_directory=parsed_args.output_directory)
    
    drone.take_off_process()
    
    initTime = time.time()
    # Start the flight
    while rclpy.ok() and not drone.landBool:
        try:
            drone.remote_control()

            if time.time() - initTime >= SAVE_FREQ: 
                drone.save_data()
                initTime = time.time()

            drone.generalTimestamps.append(time.time())

            # Process a single iteration of the ROS event loop
            rclpy.spin_once(drone, timeout_sec=VEL_PUBLISH_FREQ)

        # If any error occurs the drone lands 
        except Exception as e:
            drone.get_logger().info(str(e))
            drone.landBool = True
    
    sigint_handler(signal.SIGINT, None, drone)
    
    exit()

if __name__ == '__main__':
    main()