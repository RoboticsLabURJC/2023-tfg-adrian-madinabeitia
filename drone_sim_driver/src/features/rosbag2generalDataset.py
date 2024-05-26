#!/usr/bin/env python3
# ** Converts rosbags to raw data **

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import argparse


class rosbagDataset(Dataset):
    def __init__(self, rosbag_path) -> None:
        self.rosbag_path = rosbag_path

        self.firstVelTimestamp = -1
        self.firstImgTimestamp = -1

        self.lastVelTimestamp = 0
        self.associatedImage = False
        self.lastImgTimestamp = 0
        self.associatedVel = True

    def transform_data(self, img_topic, vel_topic):

        subdirectories = [d for d in os.listdir(self.rosbag_path) if os.path.isdir(os.path.join(self.rosbag_path, d))]

        for folderNum, subdirectory in enumerate(subdirectories):
            rosbag_dir = os.path.join(self.rosbag_path, subdirectory)

            # Create the folders if they don't exist
            img_folder_path = os.path.join(self.rosbag_path, "frontal_images")
            labels_folder_path = os.path.join(self.rosbag_path, "labels")

            os.makedirs(img_folder_path, exist_ok=True)
            os.makedirs(labels_folder_path, exist_ok=True)

            print("Folder", rosbag_dir , "as", str(folderNum))

            try:
                with ROS2Reader(rosbag_dir) as ros2_reader:

                    channels = 3  # Encoding = bgr8
                    ros2_conns = [x for x in ros2_reader.connections]
                    ros2_messages = ros2_reader.messages(connections=ros2_conns)
                    n_vel = 0
                    n_img = 0
                    self.firstVelTimestamp = -1
                    self.firstImgTimestamp = -1

                    self.lastVelTimestamp = 0
                    self.associatedImage = False
                    self.lastImgTimestamp = 0
                    self.associatedVel = True

                    # Generates all the measures of the topic
                    for m, msg in enumerate(ros2_messages):
                        (connection, timestamp, rawData) = msg

                        # Checks if it is the velocities topic
                        if connection.topic == vel_topic:
                            data = deserialize_cdr(rawData, connection.msgtype)
                            angular = data.z

                            # Conversion global frame to local frame
                            linear = data.x

                            # Checks the first timestamp
                            if self.firstVelTimestamp == -1:
                                self.firstVelTimestamp = timestamp

                            if timestamp >= self.lastVelTimestamp and not self.associatedVel:
                                # Save the data into a .txt
                                output_path = os.path.join(labels_folder_path, f"{folderNum}_{n_vel}.txt")
                                n_vel += 1
                                with open(output_path, "w") as txt_file:
                                    txt_file.write(f"{linear}, {angular}\n")
                                self.associatedImage = False

                                self.lastImgTimestamp = timestamp
                                self.associatedVel = True


                        # Checks if it is the image topic
                        if connection.topic == img_topic:
                            
                            data = deserialize_cdr(rawData, connection.msgtype)

                            # Converts the image into a readable format
                            img = np.array(data.data, dtype=data.data.dtype)
                            cvImage = img.reshape((data.height, data.width, channels))

                            # Checks the first timestamp
                            if self.firstImgTimestamp == -1:
                                self.firstImgTimestamp = timestamp

                            if timestamp >= self.lastImgTimestamp and not self.associatedImage:
                                # Save the data into a .jpg
                                output_path = os.path.join(img_folder_path, f"{folderNum}_{n_img}.jpg")
                                n_img += 1
                                resized = cv2.resize(cvImage, (160, 120))
                                cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                                
                                self.associatedVel = False

                                self.lastVelTimestamp = timestamp
                                self.associatedImage = True

            except:
                print("Folder", rosbag_dir , "is not a rosbag")


#################################################################
# Converts rosbags to general dataset                           #
#################################################################


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag data')
    parser.add_argument('--rosbags_path', type=str, help='Path to ROS bags', required=True)
    args = parser.parse_args()

    # Instantiate rosbagDataset with the provided path
    dataset = rosbagDataset(args.rosbags_path)

    img_topic = "/drone0/sensor_measurements/frontal_camera/image_raw"
    vel_topic = "/drone0/commanded_vels"    
    # img_topic = "/cf0/sensor_measurements/hd_camera/image_raw"
    # vel_topic = "commanded_vels"

    dataset.transform_data(img_topic, vel_topic)

if __name__ == "__main__":
    main()