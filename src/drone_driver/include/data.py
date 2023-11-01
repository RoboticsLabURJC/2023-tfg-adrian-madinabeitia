
#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt

import argparse
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import torch
from torch.utils.data import random_split, Dataset
import random
from collections import defaultdict

DATA_PATH = "../../rosbagsCar/datasetBag"
LOWER_LIMIT = 0
UPPER_LIMIT = 3

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.imgData = self.get_img(main_dir, "/cam_f1_left/image_raw")
        self.velData = self.get_vel(main_dir, "/cmd_vel")

        self.dataset = self.get_dataset()

    def get_dataset(self):

        return self.balanceData()

    def get_img(self, rosbag_dir, topic):
        imgs = []
        with ROS2Reader(rosbag_dir) as ros2_reader:
            
            channels = 3 # Encoding = bgr8
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)      

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)

                    # Saves the image in a readable format
                    img = np.array(data.data, dtype=data.data.dtype)
                    resizeImg = img.reshape((data.height, data.width, channels))
                    imgs.append(resizeImg)

        
        return imgs

    def get_vel(self, rosbag_dir, topic):
        vel = []

        with ROS2Reader(rosbag_dir) as ros2_reader:

            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)
            
            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    linear = data.linear.x
                    angular = data.angular.z
                    vel.append([linear, angular])
        
        return vel

    def __len__(self):
        return len(self.velData)

    def __getitem__(self, index):
        device = torch.device("cuda:0")
        image_tensor = self.transform(self.dataset[index][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[index][1]).to(device)

        return (vel_tensor, image_tensor)



    def balanceData(self):
        angular_velocities = [vel[1] for vel in self.velData]
        velocidades_lineales = [vel[0] for vel in self.velData]

        muestras_rectas = [index for index, velocidad in enumerate(angular_velocities) if abs(velocidad) <= 0.8 and abs(velocidades_lineales[index]) <= 4.5]
        muestras_curvas = [index for index, velocidad in enumerate(angular_velocities) if abs(velocidad) > 1]

        # Determina el número de muestras que deseas en la curva (4 veces más que la recta)
        num_muestras_curvas_deseado = 1000 * len(muestras_rectas)
        
        # Muestrea las muestras de recta y curva ajustando las probabilidades
        muestras_rectas_balanceadas = np.random.choice(muestras_rectas, num_muestras_curvas_deseado, replace=True)
        muestras_curvas_balanceadas = np.random.choice(muestras_curvas, num_muestras_curvas_deseado, replace=True)

        n = 5
        index_balanceado = np.concatenate([muestras_curvas_balanceadas] * n + [muestras_rectas_balanceadas])

        dataset_balanceado = [(self.imgData[i], self.velData[i]) for i in index_balanceado]

        return dataset_balanceado




def plotContinuousGraphic(label, vels, color, subplot):
    plt.subplot(2, 1, subplot)
    plt.plot(vels, label=label, linestyle='-', markersize=3, color=color)
    plt.xlabel('Muestras')
    plt.ylabel('Velocidad ' + label)
    plt.title("Velocidad " + label + " Continua")


def calculatePercentageInRange(vel, lower_limit, upper_limit):
    count_in_range = sum(1 for vel in vel if lower_limit <= vel <= upper_limit)
    total_measurements = len(vel)
    
    percentage = (count_in_range / total_measurements) * 100
    
    return percentage



dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])


#################################################################
# Data analysis for training
#################################################################


def main():
    LOWER_LIMIT = -0.25
    UPPER_LIMIT = 0.25

    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    vels = [velocitys for image, velocitys in dataset.dataset]

    linear_velocities = [vel[0] for vel in vels]
    angular_velocities = [vel[1] for vel in vels]

    # Plots the results
    plt.figure(figsize=(10, 6))

    plotContinuousGraphic("lineal", linear_velocities, 'b', 1)
    plotContinuousGraphic("angular", angular_velocities, 'g', 2)
    
    angularPerCent = calculatePercentageInRange(angular_velocities, LOWER_LIMIT, UPPER_LIMIT)
    linear_velocities = calculatePercentageInRange(linear_velocities, LOWER_LIMIT*4, UPPER_LIMIT*4)

    print(f"* Angular = 0 values => {round(angularPerCent, 2)}%")
    print(f"* Linear  = 0 values => {round(linear_velocities, 2)}%")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting