import matplotlib.pyplot as plt

import numpy as np
import sys
from ament_index_python.packages import get_package_share_directory
import argparse

package_path = get_package_share_directory("drone_behaviors")
sys.path.append(package_path)

from src.dataset.oldData import rosbagDataset, ANGULAR_UMBRALS, LINEAR_UMBRALS
from src.models.transforms import pilotNet_transforms

DATA_PATH = "../../data/outDir"

def plot_3d_bars(ax, x, y, z, xlabel='X', ylabel='Y', zlabel='Z', title='3D Plot'):

    # Adjust the heights of the bars to go from 0 to their respective z values
    width = depth = 0.15

    # Create the 3D bars with each row
    for i in range(3):
        ax.bar3d(x - width / 2 , y + i / 2 - depth / 2, np.zeros(len(y)), width, depth, z[:, i])

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)


def plot_2d(ax, vels):

    linVel, angVel = zip(*vels)
    dividedVels = []

    # Divides angular vels
    for i in range(len(angVel)):
        dividedVels.append(angVel[i] / 3)

    # Sets the labels
    ax.set_xlabel("Linear vel")
    ax.set_ylabel("Angular vel")

    ax.plot(linVel, dividedVels, linestyle='', marker='o', markersize=1)

    # Draws vertical lines to show linear umbrals
    for umbral in LINEAR_UMBRALS[:-1]:
        ax.vlines(umbral, ymin=min(dividedVels), ymax=max(dividedVels), colors='black', linestyles='dashed', linewidth=1)

    # Draws horizontal lines to show linear umbrals
    for umbral in ANGULAR_UMBRALS[:-1]:
        ax.hlines(y=umbral, xmin=min(linVel), xmax=max(linVel), colors='black', linestyles='dashed', linewidth=1)

def getLabelDistribution(labels):
    totalAngVels = np.zeros((len(LINEAR_UMBRALS), len(ANGULAR_UMBRALS)), dtype=int)

    # Counts the number of labels of each type
    for i in range(len(labels)):
        totalAngVels[labels[i][0], labels[i][1]] += 1

    return totalAngVels


def vel2label(lineal, angular):

    linLabel = 0
    angLabel = 0
    # Gets the lineal label
    if lineal < LINEAR_UMBRALS[0]:
        linLabel = 0

    for i in range(len(LINEAR_UMBRALS) - 1):
        if LINEAR_UMBRALS[i] < lineal <= LINEAR_UMBRALS[i+1]:
            linLabel = i + 1
            break

    if angular < ANGULAR_UMBRALS[0]:
        angLabel = 0

    # Gets the  angular label
    for j in range(len(ANGULAR_UMBRALS) - 1):
        if ANGULAR_UMBRALS[j] < angular <= ANGULAR_UMBRALS[j+1]:
            angLabel = j + 1
            break
    
    return [linLabel, angLabel]

def get_labels(vels):
    
    labels = []
    for vel in vels:
        labels.append(vel2label(vel[0], vel[1]/3))

    return labels


def graphData(titles, vels):
    # fig configuration
    fig = plt.figure(figsize=(12, 12))


    gs = fig.add_gridspec(2, len(vels), width_ratios=[1] * len(vels), height_ratios=[1] * 2, wspace=0.2, hspace=0.1)

    for i, vel in enumerate(vels):
        ax2D = fig.add_subplot(gs[1, i])
        ax3D = fig.add_subplot(gs[0, i], projection='3d')

        # Data for the columns 
        x = np.array([-0.5, -0.25, -0.05, 0.05, 0.25, 0.5])
        y = np.array([4, 4, 4, 4, 4, 4])

        # Gets the heights for 3D graphic
        labels = get_labels(vel)
        z = getLabelDistribution(labels)

        # 3D plot
        title_3d = titles[i] + " with " + str(len(vel)) + " samples"

        plot_2d(ax2D, vel)
        plot_3d_bars(ax3D, x, y, z.T, xlabel='Angular vel', ylabel='Linear vel', zlabel='Samples', title=title_3d)

    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag data')
    parser.add_argument('--rb', type=str, help='Path to first set of ROS bags', default=DATA_PATH)
    parser.add_argument('--rb2', type=str, help='Path to second set of ROS bags', default=None)
    parser.add_argument('--rb3', type=str, help='Path to third set of ROS bags', default=None)
    parser.add_argument('--rb4', type=str, help='Path to fourth set of ROS bags', default=None)
    args = parser.parse_args()    
    
    # Adds all the rosbags if exists 
    rosbagList = [args.rb]
    if args.rb2 is not None:
        rosbagList.append(args.rb2)

    if args.rb3 is not None:
        rosbagList.append(args.rb3)

    if args.rb4 is not None:
        rosbagList.append(args.rb4)

    # Gets the raw dataset
    data1= rosbagDataset(rosbagList, pilotNet_transforms, False, 1)
    rawVels = [velocities for image, velocities in data1.dataset]

    # Gets the balanced dataset
    balancedVels = [velocities for image, velocities in data1.balancedDataset()]

    # Gets the augmented dataset
    # data2 = rosbagDataset(DATA_PATH, dataset_transforms, False, 1)
    # augmentedVels = []
    # for i in range(len(data2)):
    #     velocity, image = data2[i]
    #     augmentedVels.append(velocity.cpu().detach().numpy())
    

    graphData(["Raw dataset", "Augmented dataset"], [rawVels, balancedVels])


if __name__ == "__main__":
    main()
