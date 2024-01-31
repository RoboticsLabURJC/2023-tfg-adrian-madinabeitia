import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ament_index_python
from imblearn.over_sampling import RandomOverSampler
import sys
from ament_index_python.packages import get_package_share_directory

package_path = get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.data import rosbagDataset, DATA_PATH, dataset_transforms


# def plot_3d_bars(x, y, z, colors, xlabel='X', ylabel='Y', zlabel='Z', title='3D Plot'):
#     # Create the figure and 3D axes
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Adjust the heights of the bars to go from 0 to their respective z values
#     bottoms = np.zeros_like(z)
#     width = depth = 0.8

#     # Create the 3D bars with different colors for each row
#     for i, color in zip(range(3), colors):
#         indices = y == i
#         ax.bar3d(x[indices], y[indices], bottoms[indices], dx=width, dy=depth, dz=z[indices], color=color)

#     # Set labels and title
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     ax.set_title(title)

#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     # Show the plot
#     plt.show()

def plot_2D_bars(x, y, colors, xLabel='X', yLabel='Y', title='3D Plot'):

    # Graphics
    colors = ['green', 'lightblue', 'blue']
    xLabel = 'Min mel    Medium vel   Max vel'
    yLabel = 'Samples'

    plt.bar(x, y, 0.1, color=colors)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)

    plt.show()


def getLabelDistribution(labels):
    positiveAngVels = [0, 0, 0]
    negativeAngVels = [0, 0, 0]

    # Counts the number of labels of each type
    for label in labels:

        if label == 3:
            positiveAngVels[2] += 1

        elif label == 2:
            positiveAngVels[1] += 1

        elif label == 1:
            positiveAngVels[0] += 1

        elif label == -3:
            negativeAngVels[2] += 1

        elif label == -2:
            negativeAngVels[1] += 1
        
        else:
            negativeAngVels[0] += 1


    return positiveAngVels, negativeAngVels



def updateNumAngular(value):
    label = 0
    if value >= 0.50:
        label = 3
    elif value >= 0.25:
        label = 2
    else:
        label = 1
    return label

def get_labels(vels):
    labels = []
    for vel in vels:
        if vel[0] >= 0:
            labels.append(updateNumAngular(vel[0]/10))
        
        else:
            labels.append(-updateNumAngular(abs(vel[0])/10))
    return labels


def main():
    # Data for the columns 
    x = np.array([-0.5, -0.25, -0.05, 0.05, 0.25, 0.5])
    
    # Gets the dataset
    data = rosbagDataset(DATA_PATH, dataset_transforms)
    vels = [velocitys for image, velocitys in data.dataset]
    labels = get_labels(vels)

    # Bars height
    nAngVelPositive, nAngVelNeagtive = getLabelDistribution(labels)
    z = np.array([nAngVelNeagtive[2], nAngVelNeagtive[1], nAngVelNeagtive[0], 
                  nAngVelPositive[0], nAngVelPositive[1], nAngVelPositive[2]])

    # Graphics
    colors = [ 'blue',]
    xLabel = 'Angular vel'
    yLabel = 'Samples'

    # plot_3d_bars(x, y, z, colors, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')
    plot_2D_bars(x, z, colors, xLabel=xLabel, yLabel=yLabel, title='Raw dataset')


    # Plots the balanced data
    dataset = data.balancedDataset()
    vels = [velocitys for image, velocitys in data.dataset]
    labels = get_labels(vels)

    # Bars height
    nAngVelPositive, nAngVelNeagtive = getLabelDistribution(labels)
    z = np.array([nAngVelNeagtive[2], nAngVelNeagtive[1], nAngVelNeagtive[0], 
                  nAngVelPositive[0], nAngVelPositive[1], nAngVelPositive[2]])


    # plot_3d_bars(x, y, z, colors, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')
    plot_2D_bars(x, z, colors, xLabel=xLabel, yLabel=yLabel, title='Balanced dataset')

if __name__ == "__main__":
    main()
