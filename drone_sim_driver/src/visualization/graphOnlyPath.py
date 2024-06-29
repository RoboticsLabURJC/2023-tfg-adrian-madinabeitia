#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


class RosbagDataset(Dataset):
    def __init__(self, rosbag_path) -> None:
        self.rosbag_path = rosbag_path

    def get_rosbag_info(self, tfTopic, velTopic):

        x_path = []
        y_path = []
        timestamps = []

        angularVels = []
        linearVels = []
        lowFilteredVels = []

        # Reads the directory
        with ROS2Reader(self.rosbag_path) as ros2_reader:
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawData) = msg

                # Search the tf topic
                if connection.topic == tfTopic:
                    data = deserialize_cdr(rawData, connection.msgtype)

                    # Appends the data in the lists
                    x_path.append(data.transforms[0].transform.translation.x)
                    y_path.append(data.transforms[0].transform.translation.y)
                    timestamps.append(timestamp)

                # Search the velocity topic
                if connection.topic == velTopic:
                    data = deserialize_cdr(rawData, connection.msgtype)
                    linearVels.append(data.x)
                    lowFilteredVels.append(data.y)
                    angularVels.append(data.z)
    
        return list(zip(x_path, y_path, timestamps)), list(zip(linearVels, lowFilteredVels, angularVels))


def plot_route(path, label):
    xPath, yPath, _ = zip(*path)
    plt.plot(xPath, yPath, linestyle='-', marker='o', markersize=0.05, label=label)

def plot_vel(vels):
    linearVel, lowFilteredVels, angularVel = zip(*vels)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # Create a figure with two subplots
    
    # Plot linear velocity
    axs[0].plot(range(len(linearVel)), linearVel, linestyle='-', markersize=0.1, label='Linear Vel', color='blue')
    axs[0].plot(range(len(lowFilteredVels)), lowFilteredVels, linestyle='-', markersize=0.1, label='FIltered vel', color='red')
    axs[0].set_title('Linear Velocity')
    
    # Plot angular velocity
    axs[1].plot(range(len(angularVel)), angularVel, linestyle='-', markersize=0.1, label='Angular vel', color='green')
    axs[1].set_title('Angular Velocity')
    
    # Optionally, you can set labels and legends
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Velocity')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')
    
    # Add a legend to the first subplot
    axs[0].legend()
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def get_lap_time(path):

    # Gets the time between each lap
    for i in range(int(len(path) / 2)):
        for j in range(int(len(path) / 2)):

            if abs(path[i][0] - path[-j][0]) < 0.1 and abs(path[i][1] - path[-j][1]) < 0.1 and abs(i - j) > 20:

                result = (path[-j][2] - path[i][2]) / 1e9
                if  result > 0:
                    return result

                else:
                    return 0
    
    return 0
def get_lap_error(refPath, path):
    """Gets the mean error in the lap

    Args:
        refPath (list[y][x]): Reference path
        path (list[y][x]): Path which the error is calculated 

    Returns:
        double: Mean error
    """
    print(len(refPath))
    slack = int(len(refPath) / 100)

    # Finds the first point of path in the reference path
    minDist = float('inf')
    index = 0

    for i in range(len(refPath)):
        # Euclidean distance
        dist = math.sqrt((refPath[i][0] - path[0][0])**2 + (refPath[i][1] - path[0][1])**2)

        if dist <= minDist:
            minDist = dist
            index = i
            indexRef = i
    
    totalErr = 0

    # Estimates the total error:
    for i in range(len(path)):
        # Corrects the trajectory
        for j in range(slack):
            ind = int((index - (slack/2 + j)))
            if ind < 0:  # Correcting index if it goes negative
                ind = len(refPath) + ind  # Wrap around to the end of refPath

            # Euclidean distance
            dist = math.sqrt((refPath[ind][0] - path[i][0])**2 + (refPath[ind][1] - path[i][1])**2)

            if dist < minDist:
                minDist = dist
                indexRef = ind

        index = indexRef
        totalErr += minDist
    
    meanError = totalErr / len(path)

    return meanError
        

def plot_doors(poses):
    for pose in poses:
        x, y, _, _, _, angle = pose
        length = 2.0  # Length of the door line
        dx = length * np.cos(angle + math.pi/2)
        dy = length * np.sin(angle + math.pi /2)
        plt.plot([x - dx / 2, x + dx / 2], [y - dy / 2, y + dy / 2], 'k-', linewidth=2)


def main(perfectPath,  filePath, doors):

    # Creates the directory if it doesn't exists 
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    # Gets the rosbags
    dataset1 = RosbagDataset(perfectPath)

    tfTopic = "/tf"
    velTopic = "/drone0/commanded_vels"

    # Gets the paths
    path1, vels1 = dataset1.get_rosbag_info(tfTopic, velTopic)

    # Prints the lapTimes
    lap1 = get_lap_time(path1)




    # Create a single figure for all plots
    label1 = 'Lap time in ' + str(round(lap1, 2)) + ' sec'



    plt.figure()

    plot_route(path1, label=label1)
    plot_doors(doors)
    plt.title('Results')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.savefig(filePath + "/route.png")


if __name__ == "__main__":

    # Gets the script arguments
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--file_path', type=str, default="./plots")

    args = parser.parse_args()

    # Doors
    poses = [

        [-6.47404, -6.98526, 0, 0, 0, -2.96],
        [-13.5561, -9.59085, 0, 0, 0, 0],
        [-18.1517, -15.3409, 0, 0, 0, 0],
        [-19.799, -25.3114, 0, 0, 0, 0],
        [-23.6129, -32.0429, 0, 0, 0, 0],
        [-31.958, -34.5389, 0, 0, 0, 0],
        [-38.6226, -31.5022, 0, 0, 0, 0],
        [-39.2746, -22.6062, 0, 0, 0, 0],
        [-37.9813, -13.5741, 0, 0, 0, 0],
        [-32.0881, -6.03558, 0, 0, 0, 0],
        [-24.4027, 0.172298, 0, 0, 0, 0],
        [-16.109, 3.8169, 0, 0, 0, 0],
        [-9.17197, 9.86202, 0, 0, 0, 0],
        [-3.33669, 16.8607, 0, 0, 0, 0],
        [2.46026, 23.7222, 0, 0, 0, 0],
        [6.6366, 33.7054, 0, 0, 0, 0],
        [13.2114, 42.9063, 0, 0, 0, 0],
        [24.221, 48.7494, 0, 0, 0, 0],
        [35.1856, 46.0491, 0, 0, 0, 0],
        [42.5478, 39.8838, 0, 0, 0, 0],
        [44.1371, 32.2892, 0, 0, 0, 0],
        [41.0288, 22.9648, 0, 0, 0, 0],
        [37.2517, 14.0092, 0, 0, 0, 0],
        [30.4535, 6.98368, 0, 0, 0, 0],
        [21.9232, 1.62634, 0, 0, 0, 0],
        [12.027, -2.23824, 0, 0, 0, -2.79],
        [2.09309, -4.03768, 0, 0, 0, -2.853]
    ]


    for i in range(1, len(poses)- 1):

        if poses[i][5] == 0:
            poses[i][5] = (np.arctan2(poses[i][1] - poses[i-1][1], poses[i][0] - poses[i-1][0]) + np.arctan2(poses[i+1][1] - poses[i][1], poses[i+1][0] - poses[i][0]))/2
        #poses[i][5] += math.pi / 4
        #poses[i][2] = random.randrange(-1, 4)


    main(args.path, args.file_path, poses)