#!/usr/bin/env python3

## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
import os
from PIL import Image
import time


DATA_PATH = "../training_dataset"
LOWER_LIMIT = 0
UPPER_LIMIT = 3

def get_image_dataset(folder_path):
    images = []

    try:
        # Lists the files
        files = os.listdir(folder_path)

        # Only reads .jpg
        jpg_files = [file for file in files if file.endswith('.jpg')]

        # Sort by timestamp
        sorted_files = sorted(jpg_files, key=lambda x: int(os.path.splitext(x)[0]))

        # Reads the images
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)

            image = Image.open(file_path)
            resized_image = image.resize((64, 64))

            image_array = np.array(resized_image)
            images.append(image_array)

    except FileNotFoundError:
        print("Error: Carpeta no encontrada.")

    return images

# Sets the label to a velocity
def updateNumAngular(value):
    label = 0
    if value > 0.50:
        label = 3

    elif value > 0.25:
        label = 2

    else:
        label = 1
    return label

def get_labels(folder_path):
    labels = []
    vels = []

    try:
        # Lists the files
        files = os.listdir(folder_path)

        # Only reads .txt
        txt_files = [file for file in files if file.endswith('.txt')]

        # Sort by timestamp
        sorted_files = sorted(txt_files, key=lambda x: int(os.path.splitext(x)[0]))

        # Reads the files
        for file_name in sorted_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                numbers = [float(num) for num in content.split(',')]

                # Updates the count
                if numbers[1] > 0:
                    labels.append(updateNumAngular(numbers[1]))
                else:
                    labels.append(-updateNumAngular(abs(numbers[1])))
                
                vels.append([numbers[1]*10])

    except FileNotFoundError:
        print("Error: Carpeta no encontrada.")
    
    return vels, labels

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        
        self.imgData = get_image_dataset(main_dir + "/frontal_images")
        self.velData, self.labels = get_labels(main_dir + "/labels")

        self.dataset = self.get_dataset()
       

    def get_dataset(self):
        
        return self.balanceData()



    def __len__(self):
        return len(self.velData)

    def __getitem__(self, item):
        device = torch.device("cuda:0")
        image_tensor = self.transform(self.dataset[item][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)

    def balanceData(self):
        balanced_dataset = [(self.imgData[i], self.velData[i]) for i in range(len(self.velData))]

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
    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    vels = [velocitys for image, velocitys in dataset.dataset]
    images = [image for image, velocitys in dataset.dataset]

    print("** Image len = ", len(images), "    Vel len = ", len(vels))




if __name__ == "__main__":
    main()
