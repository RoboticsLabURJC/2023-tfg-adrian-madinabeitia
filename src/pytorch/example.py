# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46/rosbag2_2023_10_09-11_50_46_0.db3"
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

def get_img(ros2_messages, topic):
    imgs = []
        
    for m, msg in enumerate(ros2_messages):
        (connection, timestamp, rawdata) = msg
            
        if (connection.topic == topic):
            data = deserialize_cdr(rawdata, connection.msgtype)
            imgs.append(data.data)
    
    return imgs

def get_vel(ros2_messages, topic):
    vel = []

        
    for m, msg in enumerate(ros2_messages):
        (connection, timestamp, rawdata) = msg
            
        if (connection.topic == topic):
            data = deserialize_cdr(rawdata, connection.msgtype)
            linear = data.linear.x
            angular = data.angular.z
            vel.append([linear, angular])
    
    return vel

def train(vel_data, img_data):
    # transform = transforms.Compose(
    # [transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_loader = torch.utils.data.DataLoader(img_data, batch_size=4, shuffle=True)
    print('Training set has {} instances'.format(len(img_data)))

    

def main():
    parser = argparse.ArgumentParser(description='Extract images from rosbag.')
    # input will be the folder containing the .db3 and metadata.yml file
    parser.add_argument('--input','-i',type=str, help='rosbag input location')
    

    args = parser.parse_args()
    rosbag_dir = args.input
    with ROS2Reader(rosbag_dir) as ros2_reader:

        ros2_conns = [x for x in ros2_reader.connections]
        ros2_messages = ros2_reader.messages(connections=ros2_conns)

        vel_data = get_vel(ros2_messages, "/cmd_vel")
        img_data = get_img(ros2_messages, "/filtered_img")

    print(vel_data)
    train(vel_data, img_data)
    

if __name__ == "__main__":
    main()