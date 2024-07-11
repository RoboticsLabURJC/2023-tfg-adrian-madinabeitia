import matplotlib.pyplot as plt

import numpy as np
import sys

from ament_index_python.packages import get_package_share_directory
import argparse

package_path = get_package_share_directory("drone_sim_driver")
sys.path.append(package_path)

from src.dataset.data import rosbagDataset, ALL_UMBRALS, velsDict
from src.models.transforms import deepPilot_Transforms


def get_labels(vels, spaceComponent):
    componentIndex = velsDict[spaceComponent]
    labels = [0] * len(ALL_UMBRALS[componentIndex])
    

    for vel in vels:

        linLabel = 0

        for i in range(len(ALL_UMBRALS[componentIndex]) - 1):
            if ALL_UMBRALS[componentIndex][i] < vel <= ALL_UMBRALS[componentIndex][i+1]:
                linLabel = i + 1
                break

        # Gets the lineal label
        if vel < ALL_UMBRALS[componentIndex][0]:
            linLabel = 0

        labels[linLabel] += 1

    return labels


def graphData(titles, vels, spaceComponent):
    labels = get_labels(vels, spaceComponent)

    umbrals = ALL_UMBRALS[velsDict[spaceComponent]]

    x_labels = []
    for i in range(len(labels)):
        x_labels.append( "> " + str(umbrals[i]))


    umbralsGraph = np.arange(0, len(umbrals)*2, 2)

    # Create the bar graph
    plt.bar(umbralsGraph, labels, width=[1]*len(umbralsGraph), color='skyblue')
    plt.xticks(umbralsGraph, x_labels)
    
    # Add title and labels
    plt.title(titles)
    plt.ylabel('Labels')
    plt.xlabel('Velocities')
    
    # Display the graph
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag data')
    parser.add_argument('--rb', type=str, help='Path to first set of ROS bags', required=True)
    parser.add_argument('--c', type=str, help='Extraction component', required=True)
    parser.add_argument('--rb2', type=str, help='Path to second set of ROS bags', default=None)
    parser.add_argument('--rb3', type=str, help='Path to third set of ROS bags', default=None)
    parser.add_argument('--rb4', type=str, help='Path to fourth set of ROS bags', default=None)
    args = parser.parse_args()    

    if not (args.c == "x" or args.c == "y" or args.c =="z"):
        print("You must enter --c [x or y or z]") 
        return
    
    # Adds all the rosbags if exists 
    rosbagList = [args.rb]
    if args.rb2 is not None:
        rosbagList.append(args.rb2)

    if args.rb3 is not None:
        rosbagList.append(args.rb3)

    if args.rb4 is not None:
        rosbagList.append(args.rb4)

    # Gets the raw dataset
    data1= rosbagDataset(rosbagList, deepPilot_Transforms("None"), args.c)
    rawVels = [velocities for image, velocities in data1.dataset]


    balancedVels = [velocities for image, velocities in data1.balancedDataset()]  
    # print(labels)
    
    # plt.figure()  # Create a new figure for each plot
    # plt.plot(rawVels, linestyle=' ', marker='o', markersize=1, color="Blue")
    # plt.title("a")
    # plt.xlabel('Interval Index')
    # plt.ylabel('Frequency (Hz)')
    
    # print("Mean frequency = ", np.mean(rawVels))

    # plt.show()
    title = args.c + " data: " + str(len(balancedVels))
    graphData(title, rawVels, args.c)


if __name__ == "__main__":
    main()
