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


def plot_3d_bars(x, y, z, xlabel='X', ylabel='Y', zlabel='Z', title='3D Plot'):
    # Create the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Adjust the heights of the bars to go from 0 to their respective z values
    width = depth = 0.15

    # Create the 3D bars with different colors for each row
    for i in range(3):
        ax.bar3d(x - width / 2 , y + i / 2 - depth / 2, np.zeros(len(y)), width, depth, z[:, i])

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Show the plot
    plt.show()

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
    # Crear un array NumPy inicializado a cero
    totalAngVels = np.zeros((3, 6), dtype=int)

    # Counts the number of labels of each type
    for i in range(len(labels)):
        if labels[i][0] == 3:
            totalAngVels[labels[i][1], 5] += 1

        elif labels[i][0] == 2:
            totalAngVels[labels[i][1], 4] += 1

        elif labels[i][0] == 1:
            totalAngVels[labels[i][1], 3] += 1

        elif labels[i][0] == -3:
            totalAngVels[labels[i][1], 0] += 1

        elif labels[i][0] == -2:
            totalAngVels[labels[i][1], 1] += 1
        
        else:
            totalAngVels[labels[i][1], 2] += 1

    return totalAngVels


def vel2label(lineal, angular):
    label = [0, 0]

    # Clasifies angular vels
    if angular >= 0.50:
        label[0] = 3
    elif angular >= 0.25:
        label[0] = 2
    else:
        label[0] = 1

    # Clasifies linear vels
    if lineal >= 5.5:
        label[1] = 2
    
    elif lineal >= 4:
        label[1] = 1

    else:
        label[1] = 0

    return label

def get_labels(vels):
    
    labels = []
    for vel in vels:

        if vel[1] >= 0:
            labels.append(vel2label(abs(vel[0]), abs(vel[1]/10)))
        
        else:
            lab = vel2label(abs(vel[0]), abs(vel[1])/10)
            lab[0] = -lab[0]
            labels.append(lab)

    return labels


def main():
    # Data for the columns 
    x = np.array([-0.5, -0.25, -0.05, 0.05, 0.25, 0.5])
    y = np.array([4, 4, 4, 4, 4, 4])
    
    # Gets the dataset
    data = rosbagDataset(DATA_PATH, dataset_transforms)
    vels = [velocitys for image, velocitys in data.dataset]
    labels = get_labels(vels)
    # Bars height
    z = getLabelDistribution(labels)

    print(z)
    # Graphics
    xLabel = 'Angular vel'
    yLabel = 'Linear vel'
    zLabel = 'Samples'

    plot_3d_bars(x, y, z.T, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')

    # # Plots the balanced data
    # dataset = data.balancedDataset()
    # vels = [velocitys for image, velocitys in data.dataset]
    # labels = get_labels(vels)

    # # Bars height
    # nAngVelPositive, nAngVelNeagtive = getLabelDistribution(labels)
    # z = np.array([nAngVelNeagtive[2], nAngVelNeagtive[1], nAngVelNeagtive[0], 
    #               nAngVelPositive[0], nAngVelPositive[1], nAngVelPositive[2]])


    # # plot_3d_bars(x, y, z, colors, xlabel=xLabel, ylabel=yLabel, zlabel=zLabel, title='Simple circuit')
    # plot_2D_bars(x, z, colors, xLabel=xLabel, yLabel=yLabel, title='Balanced dataset')

if __name__ == "__main__":
    main()
