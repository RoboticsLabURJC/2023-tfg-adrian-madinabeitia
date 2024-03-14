import numpy as np
import matplotlib.pyplot as plt

COLOR = ["b", "r"]

def plot_frequency(timestamps):
    frequencies = []

    for timestamp in timestamps:
        intervals = []

        for i in range(len(timestamp) - 1):
            intervals.append(1 / (timestamp[i + 1] - timestamp[i]))

        frequencies.append(intervals)

    for i, frequency in enumerate(frequencies):
        plt.figure()  # Create a new figure for each plot
        plt.plot(frequency, linestyle=' ', marker='o', markersize=3, color=COLOR[i])
        plt.title(f'Publishing Frequency - Plot {i + 1}')
        plt.xlabel('Interval Index')
        plt.ylabel('Frequency (Hz)')

    plt.show()


def main():
    #data = np.load("./sub_timestamps.npy")
    data2 = np.load("./vel_timestamps.npy")
    plot_frequency([data2])

if __name__ == "__main__":
    main()
