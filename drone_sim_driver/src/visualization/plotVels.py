import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_frequency(vels, vels2, title):


    plt.figure()  # Create a new figure for each plot
    plt.plot(vels, linestyle='-', marker='', markersize=1, color="b", label = "Commanded vel")
    plt.plot(vels2, linestyle='-', marker='', markersize=1, color="r", label="Input")
    plt.title(title)
    plt.xlabel('Timeline')
    plt.ylabel('Position')
    plt.legend()
    
    print("Mean vel = ", np.mean(vels))

    plt.show()


def main():
    # Gets the arguments
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    args = parser.parse_args()

    data1 = np.load(args.file)
    data2 = np.load(args.file2)
    plot_frequency(data1, data2, "Lineal vel")

if __name__ == "__main__":
    main()
