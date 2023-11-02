#!/bin/python3

from time import sleep
import math
import argparse
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus
from as2_python_api.modules.motion_reference_handler_module import MotionReferenceHandlerModule
from as2_motion_reference_handlers.speed_motion import SpeedMotion
from geometry_msgs.msg import TwistStamped, PoseStamped


class droneController(DroneInterface):

    def __init__(self, drone_id: str = "drone0", verbose: bool = False, use_sim_time: bool = False) -> None:
        super().__init__(drone_id, verbose, use_sim_time)
        self.motion_ref_handler = MotionReferenceHandlerModule(drone=self)


    def retry_command(self, command, check_func, sleep_time=1.0, max_retries=4):
        if not check_func():
            command()
            count = 0
            while not check_func() or count < max_retries:
                print("Retrying...")
                sleep(sleep_time)
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
            sleep(0.5)

        print("Take Off done")

    
    def land_process(self, speed):
        print("Landing")
        self.land(speed=speed)
        print("Land done")

        self.disarm()
    
    def set_vel(self, vx, vy, vz, yaw):
        velX = vx * math.cos(self.orientation[2]) + vy * math.sin(self.orientation[2])
        velY = vx * math.sin(self.orientation[2]) + vy * math.cos(self.orientation[2])

        self.motion_ref_handler.speed.send_speed_command_with_yaw_speed([velX, velY, vz], 'earth', yaw)
        
    def velocityControl(self):
        takeoff_height = 3.0
        vel = 2.0

        time_sleep = 2
        print(self.orientation)
        self.take_off_process(takeoff_height)

        self.set_vel(vel, 0, 0, 0)
        sleep(time_sleep)

        self.set_vel(0, 0, 0, 0.0)
        sleep(time_sleep)

        self.set_vel(0, 0, 0, vel)
        sleep(time_sleep)

        self.set_vel(0, 0, 0, 0.0)
        sleep(time_sleep)

        self.set_vel(0, vel, 0, 0.0)
        sleep(time_sleep)   

        self.land_process(0.5)

    def exampleMission(self):
        """ Aerostack2 with px4 example """

        speed = 0.5
        takeoff_height = 1.0
        height = 1.0

        sleep_time = 2.0

        dim = 1.0
        path = [
            [dim, dim, height],
            [dim, -dim, height],
            [-dim, dim, height],
            [-dim, -dim, height],
            [0.0, 0.0, takeoff_height],
        ]

        self.take_off_process(takeoff_height)

        ##### Mission #####
        for goal in path:
            print(f"Go to with path facing {goal}")
            self.go_to.go_to_point_path_facing(goal, speed=speed)
            print("Go to done")
        sleep(sleep_time)

        self.land_process(0.3)



if __name__ == '__main__':

    rclpy.init()

    drone = droneController(drone_id="drone0", verbose=False,
                         use_sim_time=True)
    
    drone.velocityControl()

    drone.shutdown()
    rclpy.shutdown()

    print("Clean exit")
    exit(0)
