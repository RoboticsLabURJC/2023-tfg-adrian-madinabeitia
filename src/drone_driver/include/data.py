#!/usr/bin/env python3

## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split, Dataset

import time


DATA_PATH = "../dataset/montmelo"
LOWER_LIMIT = 0
UPPER_LIMIT = 3

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform

        self.img_topic = "/drone0/sensor_measurements/frontal_camera/image_raw"
        self.vel_topic = "/drone0/motion_reference/twist"

        
        self.imgData, self.velData = self.get_from_rosbag(main_dir, self.img_topic, self.vel_topic)

        self.curveLimit = 0.5
        self.dataset = self.get_dataset()
       

    def get_dataset(self):
        return self.balanceData(self.curveLimit)

    def get_from_rosbag(self, rosbag_dir, img_topic, vel_topic):
        imgs = []
        vel = []
        initTime = time.time()


        with ROS2Reader(rosbag_dir) as ros2_reader:
            
            channels = 3 # Encoding = bgr8
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)      

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == img_topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)

                    # Saves the image in a readable format
                    img = np.array(data.data, dtype=data.data.dtype)
                    resizeImg = img.reshape((data.height, data.width, channels))
                    imgs.append(resizeImg)

                if (connection.topic == vel_topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    linear = data.twist.linear.x
                    angular = data.twist.angular.z
                    vel.append([linear, angular])       

        print("*** Data obtained in ", time.time()- initTime, " seconds ***")
        return imgs, vel



    def __len__(self):
        return len(self.velData)

    def __getitem__(self, item):
        device = torch.device("cuda:0")
        image_tensor = self.transform(self.dataset[item][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)

    def balanceData(self, angular_lim):
        curve_multiplier = 1

        print("** Image len = ", len(self.imgData), "    Vel len = ", len(self.velData))
        balanced_dataset = [(self.imgData[i], self.velData[i]) for i in range(len(self.imgData))]

        return balanced_dataset




def plotContinuousGraphic(label, vels, color, subplot):
    plt.subplot(2, 1, subplot)
    plt.plot(vels, label=label, linestyle=' ', marker='o', markersize=3, color=color)
    plt.xlabel('Sample')
    plt.ylabel('vel ' + label)
    plt.title("vel " + label)



dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])


#################################################################
# Data analysis for training                                    #
#################################################################


def main():
    D = 2  # Decimals to show in the traces
    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    vels = [velocitys for image, velocitys in dataset.dataset]
    linear_velocities = [vel[0] for vel in vels]
    angular_velocities = [vel[1] for vel in vels]

    # Plots the results
    plt.figure(figsize=(10, 6))

    plotContinuousGraphic("lineal", linear_velocities, 'b', 1)
    plotContinuousGraphic("angular", angular_velocities, 'g', 2)
    
    straight_smaples = [index for index, velocidad in enumerate(angular_velocities) if abs(velocidad) <= dataset.curveLimit]
    percentage = len(straight_smaples)/len(linear_velocities) * 1

    print(f"* Linear  samples => {round(percentage * 100, D)}%, mean = {round(np.mean(linear_velocities), D)}")
    print(f"* Curve samples => {round((1 - percentage) * 100, D)}%, mean = {round(np.mean(angular_velocities), D)}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
