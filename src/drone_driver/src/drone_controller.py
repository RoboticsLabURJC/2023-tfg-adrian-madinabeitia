#!/bin/python3

from time import sleep
import argparse
import rclpy
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg._platform_status import PlatformStatus


def retry_command(command, check_func, sleep_time=1.0, max_retries=10):
    while not check_func():
        command()
        count = 0
        while not check_func() and count < max_retries:
            print("Retrying...")
            sleep(sleep_time)
            command()
    if not check_func():
        print("Command failed")

def drone_run(drone_interface: DroneInterface):
    """ Run the mission """

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

    print("Start mission")

    ##### ARM OFFBOARD #####
    print('Offboard')
    retry_command(drone_interface.offboard, lambda: drone_interface.info['offboard'])
    print("Offboard done")

    print('Arm')
    retry_command(drone_interface.arm, lambda: drone_interface.info['armed'])
    print("Arm done")

    ##### TAKE OFF #####
    print("Take Off")
    drone_interface.takeoff(takeoff_height, speed=1.0)
    while not drone_interface.info['state'] == PlatformStatus.FLYING:
        sleep(sleep_time)
    print("Take Off done")
    

    ##### GO TO #####
    for goal in path:
        print(f"Go to with path facing {goal}")
        drone_interface.go_to.go_to_point_path_facing(goal, speed=speed)
        print("Go to done")
    sleep(sleep_time)

    ##### LAND #####
    print("Landing")
    drone_interface.land(speed=0.3)
    print("Land done")

    drone_interface.disarm()


if __name__ == '__main__':

    rclpy.init()

    uav = DroneInterface(drone_id="drone0", verbose=False,
                         use_sim_time=True)
    
    drone_run(uav)

    uav.shutdown()
    rclpy.shutdown()

    print("Clean exit")
    exit(0)
