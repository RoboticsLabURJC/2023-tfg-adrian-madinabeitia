import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import ament_index_python
from imblearn.over_sampling import RandomOverSampler
import sys
from ament_index_python.packages import get_package_share_directory

package_path = get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.data import rosbagDataset, DATA_PATH, dataset_transforms, ANGULAR_UMBRALS, LINEAR_UMBRALS



def plot_3d_bars(x, y, z, xlabel='X', ylabel='Y', zlabel='Z', title='3D Plot'):
    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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

    plt.show()


def getLabelDistribution(labels):
    totalAngVels = np.zeros((3, 6), dtype=int)

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
        labels.append(vel2label(vel[0], vel[1]/10))

    return labels


def main():
    # Data for the columns 
    x = np.array([-0.5, -0.25, -0.05, 0.05, 0.25, 0.5])
    y = np.array([4, 4, 4, 4, 4, 4])
    
    # Gets the dataset
    data = rosbagDataset(DATA_PATH, dataset_transforms)
    # vels = [velocitys for image, velocitys in data.dataset]
    # labels = get_labels(vels)
    # Bars height = Number of samples
    # z = getLabelDistribution(labels)

    # Graphics
    xLabel = 'Angular vel'
    yLabel = 'Linear vel'
    zLabel = 'Samples'

    # plot_3d_bars(x, y, z.T, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')

    # ## Plots the balanced data
    # # Plots the balanced data
    balancedDataset = data.balancedDataset()

    vels = [velocitys for image, velocitys in balancedDataset]
    labels = get_labels(vels)

    
    # Bars height = Number of samples
    z = getLabelDistribution(labels)

    plot_3d_bars(x, y, z.T, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')

if __name__ == "__main__":
    main()
