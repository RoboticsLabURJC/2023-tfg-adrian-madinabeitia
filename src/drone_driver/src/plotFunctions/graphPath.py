#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
from torch.utils.data import Dataset
import os
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

        return x_path, y_path, timestamps


def plot_route(x_paths, y_paths, label):
    plt.plot(x_paths, y_paths, linestyle='-', marker='o', markersize=0.1, label=label)


def printLapTime(xPath, yPath, timestamps):
    # Gets the time between each lap
    for i in range(int(len(xPath) / 4)):
        for j in range(int(len(xPath) / 4)):

            if abs(xPath[i] - xPath[-j]) < 0.1 and abs(yPath[i] - yPath[-j]) < 0.1 and abs(i - j) > 20:

                result = (timestamps[-j] - timestamps[i]) / 1e9
                if  result > 0:
                    return result

                else:
                    return 0
    
    return 0

def main(path3):

    # Gets the rosbags
    dataset_1 = RosbagDataset(ROSBAGS_PATH_1)
    dataset_2 = RosbagDataset(ROSBAGS_PATH_2)
    dataset_3 = RosbagDataset(path3)

    tf_topic = "/tf"

    # Gets the paths
    x_path_1, y_path_1, timeStamps1 = dataset_1.get_path(tf_topic)
    x_path_2, y_path_2, timeStamps2 = dataset_2.get_path(tf_topic)
    x_path_3, y_path_3, timeStamps3 = dataset_3.get_path(tf_topic)

    # Prints the lapTimes
    lap1 = printLapTime(x_path_1, y_path_1, timeStamps1)
    lap2 = printLapTime(x_path_2, y_path_2, timeStamps2)
    lap3 = printLapTime(x_path_3, y_path_3, timeStamps3)

    # Create a single figure for all plots
    label1 = 'Slow expert pilot in ' + str(round(lap1, 2)) + ' sec'
    label2 = 'Fast expert pilot in ' + str(round(lap2, 2)) + ' sec'
    label3 = 'Fast expert pilot in ' + str(round(lap3, 2)) + ' sec'

    plt.figure()

    plot_route(x_path_1, y_path_1, label=label1)
    plot_route(x_path_2, y_path_2, label=label2)
    plot_route(x_path_3, y_path_3, label=label3)

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