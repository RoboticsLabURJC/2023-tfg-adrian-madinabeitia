import numpy as np
import matplotlib.pyplot as plt

def plot_frequency(timestamps):
    intervals = []

    for timestamp in timestamps:
        interv = []

        for i in range(len(timestamp) - 1):
            interv.append(timestamp[i + 1] - timestamp[i])

        intervals.append(interv)

    for i, interval in enumerate(intervals):
        plt.figure()  # Crea una nueva figura para cada plot
        plt.plot(interval)
        plt.title(f'Timestamp Intervals - Plot {i + 1}')
        plt.xlabel('Interval Index')
        plt.ylabel('Time Interval (Miliseconds)')

    plt.show() 


def main():
    data = np.load("./TimestampsData.txt")
    plot_frequency(data)

if __name__ == "__main__":
    main()