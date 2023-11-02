
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


DATA_PATH = "../../rosbagsCar/datasetBag"
LOWER_LIMIT = 0
UPPER_LIMIT = 3

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.imgData = self.get_img(main_dir, "/cam_f1_left/image_raw")
        self.velData = self.get_vel(main_dir, "/cmd_vel")

        self.curveLimit = 0.5
        self.dataset = self.get_dataset()
       

    def get_dataset(self):
        return self.balanceData(self.curveLimit)

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

    def __getitem__(self, item):
        device = torch.device("cuda:0")
        image_tensor = self.transform(self.dataset[item][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)

    def balanceData(self, angular_lim):
        curve_multiplier = 1

        angular_velocities = [vel[1] for vel in self.velData]

        straight_smaple = [index for index, vel in enumerate(angular_velocities) if abs(vel) <= angular_lim]
        curve_sample = [index for index, vel in enumerate(angular_velocities) if abs(vel) > angular_lim]
        
        # Balances the numumber of curves in the dataset
        curve_aument = int(1 / (len(curve_sample) / len(self.velData)))
        n_curve = curve_multiplier * curve_aument   # Inreases the curve samples

        balanced_index = np.concatenate([curve_sample] * n_curve + [straight_smaple])
        balanced_dataset = [(self.imgData[i], self.velData[i]) for i in balanced_index]

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



    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting