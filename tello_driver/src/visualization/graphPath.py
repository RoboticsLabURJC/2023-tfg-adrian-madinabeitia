#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import argparse
import os


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
    plt.plot(xPath, yPath, linestyle='-', marker='o', markersize=0.1, label=label)

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
        


def main(perfectPath, expertPath, neuralPath, filePath, traceBool):

    # Creates the directory if it doesn't exists 
    if not os.path.exists(filePath):
        os.makedirs(filePath)

    # Gets the rosbags
    dataset1 = RosbagDataset(perfectPath)
    dataset2 = RosbagDataset(expertPath)
    dataset3 = RosbagDataset(neuralPath)

    tfTopic = "/tf"
    velTopic = "/drone0/commanded_vels"

    # Gets the paths
    path1, vels1 = dataset1.get_rosbag_info(tfTopic, velTopic)
    path2, vels2 = dataset2.get_rosbag_info(tfTopic, velTopic)
    path3, vels3 = dataset3.get_rosbag_info(tfTopic, velTopic)

    # Prints the lapTimes
    lap1 = get_lap_time(path1)
    lap2 = get_lap_time(path2)
    lap3 = get_lap_time(path3)

    # Calculates the error between trajectories
    error1 = get_lap_error(path2, path1)
    error2 = get_lap_error(path3,  path1)

    # Create a single figure for all plots
    label1 = 'Slow expert pilot in ' + str(round(lap1, 2)) + ' sec'
    label2 = 'Fast expert pilot in ' + str(round(lap2, 2)) + ' sec | Error = ' + str(round(error1, 2)) + 'm'
    label3 = 'Neural pilot in ' + str(round(lap3, 2)) + ' sec | Error = ' + str(round(error2, 2)) + 'm'

    if traceBool:
        plot_vel(vels3)

    plt.figure()

    plot_route(path1, label=label1)
    plot_route(path2, label=label2)
    plot_route(path3, label=label3)

    plt.title('Results')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.savefig(filePath + "/route.png")


if __name__ == "__main__":

    # Gets the script arguments
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--perfect_path', type=str, required=True)
    parser.add_argument('--expert_path', type=str, required=True)
    parser.add_argument('--neural_path', type=str, required=True)
    parser.add_argument('--file_path', type=str, default="./plots")
    parser.add_argument('--show_vels', type=str, help='Show the traces')

    args = parser.parse_args()

    # Checks the bool argument
    traceBool = False
    if args.show_vels == "True" or args.show_vels == "true":
        traceBool = True

    main(args.perfect_path, args.expert_path, args.neural_path, args.file_path, traceBool)