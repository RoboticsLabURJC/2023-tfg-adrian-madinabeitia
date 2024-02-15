#!/usr/bin/env python3


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

ANGULAR_UMBRALS = [-1.5, -0.6, -0.3, 0, 0.3, 0.6, 1.5, float('inf')]  # label < umbral
LINEAR_UMBRALS = [5, 5.5, float('inf')]

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
        self.minSamples = 25
        
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

    def getLinearSubset(self, dataset, lowerBound, upperBound):
        linearSub = [(img, vel) for img, vel in dataset if lowerBound < vel[0] <= upperBound]

        return linearSub
    
    def getAngularSubset(self, dataset, lowerBound, upperBound):
        angularSub = [(img, vel) for img, vel in dataset if lowerBound < vel[1]/10 <= upperBound]

        # Gets the linear subsets
        angularLabeled = [self.getLinearSubset(angularSub, float('-inf'), LINEAR_UMBRALS[0])]

        for i in range(0, len(LINEAR_UMBRALS) - 1):
            angularLabeled.append(self.getLinearSubset(angularSub, LINEAR_UMBRALS[i], LINEAR_UMBRALS[i+1]))

        return angularLabeled

    
    def balancedDataset(self):
        #   <       5     5.5   inf
        weights = [(0.40, 0.0, 0.0),     # < -1.5
                   (0.50, 0.3, 0.0),     # < -0.6
                   (0.60, 0.5, 0.7),     # < -0.3
                   (0.70, 0.8, 0.99),    # < 0
                   (0.70, 0.8, 0.99),    # < 0.3
                   (0.60, 0.5, 0.7),     # < 0.6
                   (0.50, 0.3, 0.0),     # < 1.5
                   (0.40, 0.0, 0.0)]     # < inf

        # weights = [(0.99, 0.99, 0.99),
        #            (0.99, 0.99, 0.99),     # < -0.5
        #            (0.99, 0.99, 0.99),     # < -0.25
        #            (0.99, 0.99, 0.99),     # < 0
        #            (0.99, 0.99, 0.99),     # < 0.25
        #            (0.99, 0.99, 0.99),     # < 0.5
        #            (0.99, 0.99, 0.99),
        #            (0.99, 0.99, 0.99)]     # < inf

        # Gets all the subsets
        labeledDataset = [self.getAngularSubset(self.dataset, float('-inf'), ANGULAR_UMBRALS[0])]
        
        for i in range(0, len(ANGULAR_UMBRALS) - 1):
            labeledDataset.append(self.getAngularSubset(self.dataset, ANGULAR_UMBRALS[i], ANGULAR_UMBRALS[i+1]))

        # Gets the max number of samples in a label
        maxSamples = 0
        
        for angLabeled in labeledDataset:
            for linLabeled in angLabeled:
                if len(linLabeled) > maxSamples:
                    maxSamples = len(linLabeled)

        # Balances all the clases
        balancedDataset = []

        for i, angLabeled in enumerate(labeledDataset):
            balancedAngLabeled = []
            
            for j in range(len(angLabeled)):
                currentSubset = angLabeled[j]
                currentSubsetLen = len(currentSubset)
                
                if currentSubsetLen >= self.minSamples:

                    # Checks if thers enought samples
                    if currentSubsetLen < maxSamples:

                        # If the current length is less than maxSamples, we replicate the elements
                        repetitions = maxSamples // currentSubsetLen
                        remainder = maxSamples % currentSubsetLen
                        balancedSubset = currentSubset * repetitions + currentSubset[:remainder]

                    else:
                        # If the current length is greater than or equal to maxSamples, we subsample
                        balancedSubset = random.sample(currentSubset, maxSamples)
                
                    # Adjusts to weights
                    balancedSubset = self.subSample(balancedSubset, 1 - weights[i][j])
                    balancedAngLabeled.append(balancedSubset)
            
            balancedDataset.append(balancedAngLabeled)

        # Joins all the labels
        rawDataset = []

        for angLabeled in balancedDataset:
            for linLabeled in angLabeled:
                for data in linLabeled:
                    rawDataset.append(data)


        return rawDataset


dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])
