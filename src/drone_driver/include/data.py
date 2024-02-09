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
import random
from torch.utils.data import Dataset
import os
from PIL import Image
import time
from itertools import cycle, islice


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



def get_vels(folder_path):
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

                vels.append([numbers[0], numbers[1]*10])

    except FileNotFoundError:
        print("Error: Carpeta no encontrada.")
    
    return vels

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        
        self.imgData = get_image_dataset(main_dir + "/frontal_images")
        self.velData = get_vels(main_dir + "/labels")

        self.dataset = [(self.imgData[i], self.velData[i]) for i in range(len(self.velData))]
       




    def __len__(self):
        return len(self.velData)

    def __getitem__(self, item):
        device = torch.device("cuda:0")
        image_tensor = self.transform(self.dataset[item][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)
    

    def subSample(self, label, percent):
        toDel = int(len(label) * percent)
        subsampled = random.sample(label, len(label) - toDel)

        return subsampled
    
    def getSubset(self, dataset, lower_bound, upper_bound):
        angularSub = [(img, vel) for img, vel in dataset if lower_bound <= vel[1]/10 < upper_bound]
        
        linear1 = [(img, vel) for img, vel in angularSub if vel[0] < 4.5]
        linear2 = [(img, vel) for img, vel in angularSub if 4.5 <= vel[0] < 5]
        linear3 = [(img, vel) for img, vel in angularSub if 5 <= vel[0] < 6]
        print(len(angularSub), len(linear1), len(linear2), len(linear3))
        return [linear1, linear2, linear3]

    def oversample(self, label, max_count):
        return list(islice(cycle(label), max_count))
    
    def balancedDataset(self):
        balanced_dataset = []
        #          -0.5  -0.25  -0.0   0.0  0.25   0.5    
        weights = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),    # 4.0
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),    # 4.5
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]    # 5.0


        # Gets all the subsets
        label3 = self.getSubset(self.dataset, 0.5, float('inf'))
        label2 = self.getSubset(self.dataset, 0.25, 0.5)
        label1 = self.getSubset(self.dataset, 0.0, 0.25)
        label_1 = self.getSubset(self.dataset, -0.25, 0.0)
        label_2 = self.getSubset(self.dataset, -0.5, -0.25)
        label_3 = self.getSubset(self.dataset, float('-inf'), -0.5)

        # Balances all the clases
        for i in range(3):
            max_count = max(len(label3[i]), len(label2[i]), len(label1[i]), 
                            len(label_1[i]), len(label_2[i]), len(label_3[i]))
            
            label3_oversampled = self.oversample(label3[i], max_count)
            label2_oversampled = self.oversample(label2[i], max_count)
            label1_oversampled = self.oversample(label1[i], max_count)
            label_1_oversampled = self.oversample(label_1[i], max_count)
            label_2_oversampled = self.oversample(label_2[i], max_count)
            label_3_oversampled = self.oversample(label_3[i], max_count)

            # Subsamples and adjust with the weights all the clases
            label_3_oversampled = self.subSample(label_3_oversampled, weights[i][0])
            label_2_oversampled = self.subSample(label_2_oversampled, weights[i][1])
            label_1_oversampled = self.subSample(label_1_oversampled, weights[i][2])
            label1_oversampled = self.subSample(label1_oversampled, weights[i][3])
            label2_oversampled = self.subSample(label2_oversampled, weights[i][4])
            label3_oversampled = self.subSample(label3_oversampled, weights[i][5])

            # Crear un conjunto de datos balanceado
            balancedSubSet = (
                label3_oversampled + label2_oversampled + label1_oversampled +
                label_1_oversampled + label_2_oversampled + label_3_oversampled
            )
            balanced_dataset.extend(balancedSubSet)
        self.dataset = balanced_dataset

        return balanced_dataset


dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])
