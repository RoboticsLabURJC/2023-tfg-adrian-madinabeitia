#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np

RED= 1

def get_rosbag_info(rosbag_path, tfTopic, velTopic):
    x_path = []
    y_path = []
    z_path = []
    timestamps = []

    angularVels = []
    linearVels = []
    lowFilteredVels = []

    with ROS2Reader(rosbag_path) as ros2_reader:
        ros2_conns = [x for x in ros2_reader.connections]
        ros2_messages = ros2_reader.messages(connections=ros2_conns)

        for m, msg in enumerate(ros2_messages):
            (connection, timestamp, rawData) = msg

            if connection.topic == tfTopic:
                data = deserialize_cdr(rawData, connection.msgtype)
                x_path.append(data.transforms[0].transform.translation.x/RED)
                y_path.append(data.transforms[0].transform.translation.y/RED)
                z_path.append(data.transforms[0].transform.translation.z)
                timestamps.append(timestamp)

            if connection.topic == velTopic:
                data = deserialize_cdr(rawData, connection.msgtype)
                linearVels.append(data.x)
                lowFilteredVels.append(data.y)
                angularVels.append(data.z)

    return list(zip(x_path, y_path, z_path, timestamps)), list(zip(linearVels, lowFilteredVels, angularVels))


def plot_route_3d(path, label, doors):
    xPath, yPath, zPath, _ = zip(*path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xPath, yPath, zPath, _ = zip(*path)
    ax.plot(xPath, yPath, zPath, linestyle='-', marker='o', markersize=2, label=label)

    plot_doors_3d(ax, doors)
    ax.set_title('3D Route with Doors')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(0, 10)

    ax.grid(True, linestyle='--', linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_doors_3d(ax, poses):
    for pose in poses:
        x, y, z, _, _, angle = pose
        length = 8.0  # Length of the door line
        height = 1.5  # Height of the door
        pole_height = z - (-3)  # Height of the pole
        
        # Calculate door direction
        dx = length * np.cos(angle + np.pi/2)
        dy = length * np.sin(angle + np.pi/2)
        
        # Coordinates of the square (door)
        square_x = [x - dx / 2, x + dx / 2, x + dx / 2, x - dx / 2, x - dx / 2]
        square_y = [y - dy / 2, y + dy / 2, y + dy / 2, y - dy / 2, y - dy / 2]
        square_z = [z, z, z + height, z + height, z]
        
        # Plot the door (square)
        ax.plot(square_x, square_y, square_z, 'k-', linewidth=4)
        
        # Coordinates for the pole
        pole_x = [x, x]
        pole_y = [y, y]
        pole_z = [z, 0]
        
        # Plot the pole
        ax.plot(pole_x, pole_y, pole_z, 'k-', linewidth=2)



def get_lap_time(path):
    for i in range(int(len(path) / 2)):
        for j in range(int(len(path) / 2)):
            if abs(path[i][0] - path[-j][0]) < 0.1 and abs(path[i][1] - path[-j][1]) < 0.1 and abs(i - j) > 20:
                result = (path[-j][2] - path[i][2]) / 1e9
                if result > 0:
                    return result
                else:
                    return 0
    return 0


def get_lap_error(refPath, path):
    print(len(refPath))
    slack = int(len(refPath) / 100)
    minDist = float('inf')
    index = 0

    for i in range(len(refPath)):
        dist = math.sqrt((refPath[i][0] - path[0][0])**2 + (refPath[i][1] - path[0][1])**2)
        if dist <= minDist:
            minDist = dist
            index = i
            indexRef = i

    totalErr = 0
    for i in range(len(path)):
        for j in range(slack):
            ind = int((index - (slack / 2 + j)))
            if ind < 0:
                ind = len(refPath) + ind
            dist = math.sqrt((refPath[ind][0] - path[i][0])**2 + (refPath[ind][1] - path[i][1])**2)
            if dist < minDist:
                minDist = dist
                indexRef = ind
        index = indexRef
        totalErr += minDist

    meanError = totalErr / len(path)
    return meanError


def main(perfectPath, doors):
    tfTopic = "/tf"
    velTopic = "/drone0/commanded_vels"
    path1, vels1 = get_rosbag_info(perfectPath, tfTopic, velTopic)

    lap1 = get_lap_time(path1)
    label1 = 'Lap time in ' + str(round(lap1, 2)) + ' sec'

    plot_route_3d(path1, label1, doors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    poses = [
            [-6.47404, -6.98526, 2.7153, 0, 0, -2.96],
            [-13.5561, -9.59085, -0.4432, 0, 0, 0],
            [-18.1517, -15.3409, 1.2568, 0, 0, 0],
            [-19.799, -25.3114, -0.3211, 0, 0, 0],
            [-23.6129, -32.0429, 0.7894, 0, 0, 0],
            [-31.958, -34.5389, 2.4873, 0, 0, 0],
            [-38.6226, -31.5022, 1.9345, 0, 0, 0],
            [-39.2746, -22.6062, 0.5432, 0, 0, 0],
            [-37.9813, -13.5741, 1.8765, 0, 0, 0],
            [-32.0881, -6.03558, -0.1245, 0, 0, 0],
            [-24.4027, 0.172298, 2.1534, 0, 0, 0],
            [-16.109, 3.8169, 1.4532, 0, 0, 0],
            [-9.17197, 9.86202, -0.3214, 0, 0, 0],
            [-3.33669, 16.8607, 0.8765, 0, 0, 0],
            [2.46026, 23.7222, 2.6543, 0, 0, 0],
            [6.6366, 33.7054, 1.1234, 0, 0, 0],
            [13.2114, 42.9063, -0.9876, 0, 0, 0],
            [24.221, 48.7494, 1.2345, 0, 0, 0],
            [35.1856, 46.0491, 0.5432, 0, 0, 0],
            [42.5478, 39.8838, 1.8765, 0, 0, 0],
            [44.1371, 32.2892, -0.1245, 0, 0, 0],
            [41.0288, 22.9648, 2.1534, 0, 0, 0],
            [37.2517, 14.0092, 1.4532, 0, 0, 0],
            [30.4535, 6.98368, -0.3214, 0, 0, 0],
            [21.9232, 1.62634, 0.8765, 0, 0, 0],
            [12.027, -2.23824, 2.6543, 0, 0, -2.79],
            [2.09309, -4.03768, 3.1234, 0, 0, -2.853]
    ]

    for i in range(1, len(poses)-1):
        poses[i][0] = poses[i][0]/RED
        poses[i][1] = poses[i][1]/RED
        #poses[i][2] = 0.0
        poses[i][2] += 1.65
        poses[i][2] = poses[i][2]

        #if poses[i][5] == 0:
        poses[i][5] = ((np.arctan2(poses[i][1] - poses[i-1][1], poses[i][0] - poses[i-1][0]) + np.arctan2(poses[i+1][1] - poses[i][1], poses[i+1][0] - poses[i][0]))/2) #- math.pi

    poses[0][2] += 1.65

    main(args.path, poses)
