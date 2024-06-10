#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import random
import albumentations as A
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import os
from PIL import Image



# Labels

# Gates extended
ANGULAR_UMBRALS = [-0.35, -0.10, 0, 0.10, 0.35, float('inf')]
LINEAR_UMBRALS = [2.4, 3.0, float('inf')]

# General aspects
USE_WEIGHTS = True
USE_AUGMENTATION = False

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
        print("Error: File not found.")

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
        
                vels.append([abs(numbers[0]), numbers[1]*3])

    except FileNotFoundError:
        print("Error: File not found.")
    
    return vels



class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.minSamples = 2
        
        self.imgData = []
        self.velData = []
        for dir in main_dir:
            self.imgData.extend(get_image_dataset(dir + "/frontal_images"))
            self.velData.extend(get_vels(dir + "/labels"))

        self.dataset =  self.get_dataset() 
        

    def __len__(self):
        return len(self.dataset)

    def get_dataset(self):
        dataset = list(((self.imgData[i], self.velData[i]) for i in range(len(self.velData))))

        original_size = len(dataset)
        new_size = int(original_size * self.dataAugment)

        for _ in range(new_size - original_size):
            randIndex = random.randint(0, original_size - 1)
            dataset.append((self.imgData[randIndex], self.velData[randIndex]))
        
        return dataset


    def __getitem__(self, item):
        device = torch.device("cuda:0")
        
        # angular = ( + MAX_ANGULAR) / (2 * MAX_ANGULAR)
        # lineal = ( + MIN_LINEAR) / (MAX_LINEAR - MIN_LINEAR)

        angular = self.dataset[item][1][1] * 3
        lineal = self.dataset[item][1][0]
        vels = (lineal, angular)


        image_tensor = self.transform(self.dataset[item][0])
        vel_tensor = torch.tensor(vels).to(device)

            
        return (vel_tensor, image_tensor)    
        

    def subSample(self, label, percent):
        toDel = int(len(label) * percent)
        subsample = random.sample(label, len(label) - toDel)

        return subsample

    def getLinearSubset(self, dataset, lowerBound, upperBound):
        linearSub = [(img, vel) for img, vel in dataset if lowerBound < vel[0] <= upperBound]

        return linearSub
    
    def getAngularSubset(self, dataset, lowerBound, upperBound):
        angularSub = [(img, vel) for img, vel in dataset if lowerBound < vel[1]/3 <= upperBound]

        # Gets the linear subsets
        angularLabeled = [self.getLinearSubset(angularSub, float('-inf'), LINEAR_UMBRALS[0])]

        for i in range(0, len(LINEAR_UMBRALS) - 1):
            angularLabeled.append(self.getLinearSubset(angularSub, LINEAR_UMBRALS[i], LINEAR_UMBRALS[i+1]))

        return angularLabeled

    
    def balancedDataset(self):
        useWeights = USE_WEIGHTS

        # Combined dataset
        weights = [(0.7, 0.65, 0.5),     # < -0.6
                   (0.75, 0.85, 0.75),     # < -0.3
                   (0.70, 0.90, 0.80),    # < 0
                   (0.70, 0.90, 0.80),    # < 0.3
                   (0.75, 0.85, 0.75),     # < 0.6
                   (0.7, 0.65, 0.5)]      # < inf
                

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

        # Augments the data
        maxSamples = int(maxSamples * self.dataAugment)        
        
        # Balances all the classes
        balancedDataset = []

        for i, angLabeled in enumerate(labeledDataset):
            balancedAngLabeled = []
            
            for j in range(len(angLabeled)):
                currentSubset = angLabeled[j]
                currentSubsetLen = len(currentSubset)
                
                if currentSubsetLen >= self.minSamples:

                    # Checks if there's enough samples
                    if currentSubsetLen < maxSamples:

                        # If the current length is less than maxSamples, we replicate the elements
                        repetitions = maxSamples // currentSubsetLen
                        remainder = maxSamples % currentSubsetLen
                        balancedSubset = currentSubset * repetitions + currentSubset[:remainder]

                    else:
                        # If the current length is greater than or equal to maxSamples, we subsample
                        balancedSubset = random.sample(currentSubset, maxSamples)
                    
                    # Subsample
                    if useWeights:
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

        self.dataset = rawDataset

        return rawDataset

