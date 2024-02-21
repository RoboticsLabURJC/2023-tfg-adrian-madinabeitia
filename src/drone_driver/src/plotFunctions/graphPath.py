#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
from torch.utils.data import Dataset
import math
import matplotlib.pyplot as plt
import argparse

ROSBAGS_PATH_1 = "./positionRecords/perfectNurburGingTF"
ROSBAGS_PATH_2 = "./positionRecords/EPNumbirngTF"
ROSBAGS_PATH_3 = "./positionRecords/"

class RosbagDataset(Dataset):
    def __init__(self, rosbag_path) -> None:
        self.rosbag_path = rosbag_path

    def get_path(self, tf_topic):
        x_path = []
        y_path = []
        timestamps = []

        # Reads the directory
        with ROS2Reader(self.rosbag_path) as ros2_reader:
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg

                # Searchs the tf topic
                if connection.topic == tf_topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)

                    # Appends the data in the lists
                    x_path.append(data.transforms[0].transform.translation.x)
                    y_path.append(data.transforms[0].transform.translation.y)
                    timestamps.append(timestamp)

        return list(zip(x_path, y_path, timestamps))


def plot_route(path, label):
    xPath, yPath, _ = zip(*path)
    plt.plot(xPath, yPath, linestyle='-', marker='o', markersize=0.1, label=label)


def get_lap_time(path):
    # Gets the time between each lap
    for i in range(int(len(path) / 4)):
        for j in range(int(len(path) / 4)):

            if abs(path[i][0] - path[-j][0]) < 0.1 and abs(path[i][1] - path[-j][1]) < 0.1 and abs(i - j) > 20:

                result = (path[-j][2] - path[i][2]) / 1e9
                if  result > 0:
                    return result

                else:
                    return 0
    
    return 0

def get_lap_error(refPath, path):
    slack = 5

    # Finds the first point of path in the reference path
    minDist = float('inf')
    index = 0

    for i in range(len(refPath)):
        # Euclidean distance
        dist = math.sqrt((refPath[i][0] - path[0][0])**2 + (refPath[i][1] - path[0][1])**2)

        if dist < minDist:
            minDist = dist
            index = i
    
    totalErr = 0

    # Estimates the total error:
    for i in range(len(path)):
        # Corrects the trajectory
        for j in range(slack):
            ind = (index + j) % len(refPath)

            # Euclidean distance
            dist = math.sqrt((refPath[ind][0] - path[i][0])**2 + (refPath[ind][1] - path[i][1])**2)

            if dist < minDist:
                minDist = dist
                index = ind
        
        totalErr += minDist
    
    meanError = totalErr / len(path)

    return meanError
        


def main(path3):

    # Gets the rosbags
    dataset1 = RosbagDataset(ROSBAGS_PATH_1)
    dataset2 = RosbagDataset(ROSBAGS_PATH_2)
    dataset3 = RosbagDataset(path3)

    tfTopic = "/tf"

    # Gets the paths
    path1 = dataset1.get_path(tfTopic)
    path2 = dataset2.get_path(tfTopic)
    path3 = dataset3.get_path(tfTopic)

    # Prints the lapTimes
    lap1 = get_lap_time(path1)
    lap2 = get_lap_time(path2)
    lap3 = get_lap_time(path3)

    # Calculates the error between trajectories
    error1 = get_lap_error(path1, path2) * 10
    error2 = get_lap_error(path1,  path3) * 10

    # Create a single figure for all plots
    label1 = 'Slow expert pilot in ' + str(round(lap1, 2)) + ' sec'
    label2 = 'Fast expert pilot in ' + str(round(lap2, 2)) + ' sec | Error = ' + str(round(error1, 2))
    label3 = 'Neural pilot in ' + str(round(lap3, 2)) + ' sec | Error = ' + str(round(error2, 2))

    plt.figure()

    plot_route(path1, label=label1)
    plot_route(path2, label=label2)
    plot_route(path3, label=label3)

    plt.title('Results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--p', type=str, default="Balanced2",
                        help='Path to the third ROS bag dataset')

    args = parser.parse_args()
    main(ROSBAGS_PATH_3 + args.p)