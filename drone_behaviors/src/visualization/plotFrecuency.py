import numpy as np
import matplotlib.pyplot as plt
import argparse

COLOR = ["b", "r"]

def plot_frequency(timestamps, title):
    frequencies = []

    for timestamp in timestamps:
        intervals = []

        for i in range(len(timestamp) - 1):
            intervals.append(1 / (timestamp[i + 1] - timestamp[i]))

        frequencies.append(intervals)

    for i, frequency in enumerate(frequencies):
        plt.figure()  # Create a new figure for each plot
        plt.plot(frequency, linestyle=' ', marker='o', markersize=1, color=COLOR[i])
        plt.title(title)
        plt.xlabel('Interval Index')
        plt.ylabel('Frequency (Hz)')
    
    print("Mean frequency = ", np.mean(frequencies))

    plt.show()


def main():
    # Gets the arguments
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.file)
    plot_frequency([data], "Freq")

if __name__ == "__main__":
    main()
