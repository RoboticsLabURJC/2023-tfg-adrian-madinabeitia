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
import os
import time
import cv2


ROSBAGS_PATH = "../dataset"

class rosbagDataset(Dataset):
    def __init__(self, rosbag_path) -> None:
        self.rosbag_path = rosbag_path

        self.firstVelTimestamp = -1
        self.firstImgTimestamp = -1

        self.lastVelTimestamp = 0
        self.asocciatedImage = True
       

    def transform_data(self, img_topic, vel_topic):

        subdirectories = [d for d in os.listdir(self.rosbag_path) if os.path.isdir(os.path.join(self.rosbag_path, d))]

        for subdirectory in subdirectories:
            rosbag_dir = os.path.join(self.rosbag_path, subdirectory)

            # Create the folders if they don't exist
            img_folder_path = os.path.join(rosbag_dir, "frontal_images")
            labels_folder_path = os.path.join(rosbag_dir, "labels")

            os.makedirs(img_folder_path, exist_ok=True)
            os.makedirs(labels_folder_path, exist_ok=True)

            with ROS2Reader(rosbag_dir) as ros2_reader:

                channels = 3  # Encoding = bgr8
                ros2_conns = [x for x in ros2_reader.connections]
                ros2_messages = ros2_reader.messages(connections=ros2_conns)


                # Generates all the mesaures of the topic
                for m, msg in enumerate(ros2_messages):
                    (connection, timestamp, rawdata) = msg

                    # Checks if is the velocitys topic
                    if connection.topic == vel_topic:
                        data = deserialize_cdr(rawdata, connection.msgtype)
                        linear = data.twist.linear.x
                        angular = data.twist.angular.z
                        

                        # Checks the first timestamp
                        if self.firstVelTimestamp == -1:
                            self.firstVelTimestamp = timestamp

                        # Save the data into a .txt
                        output_path = os.path.join(labels_folder_path, f"{timestamp - self.firstVelTimestamp}.txt")
                        with open(output_path, "w") as txt_file:
                            txt_file.write(f"{linear}, {angular}\n")
                        
                        self.lastVelTimestamp = timestamp
                        self.asocciatedImage = False

                    # Checks if is the image topic
                    if connection.topic == img_topic:
                        data = deserialize_cdr(rawdata, connection.msgtype)

                        # Converts the image in a readable format
                        img = np.array(data.data, dtype=data.data.dtype)
                        resizeImg = img.reshape((data.height, data.width, channels))

                        # Checks the first timestamp
                        if self.firstImgTimestamp == -1:
                            self.firstImgTimestamp = timestamp

                        if timestamp >= self.lastVelTimestamp and not self.asocciatedImage:
                            # Save the data into a .jpg
                            output_path = os.path.join(img_folder_path, f"{timestamp - self.firstImgTimestamp}.jpg")
                            cv2.imwrite(output_path, cv2.cvtColor(resizeImg, cv2.COLOR_BGR2RGB))
                            
                            self.asocciatedImage = True




#################################################################
# Converts rosbags to general dataset                           #
#################################################################


def main():
    dataset = rosbagDataset(ROSBAGS_PATH)

    img_topic = "/drone0/sensor_measurements/frontal_camera/image_raw"
    vel_topic = "/drone0/motion_reference/twist"

    dataset.transform_data(img_topic, vel_topic)


if __name__ == "__main__":
    main()
