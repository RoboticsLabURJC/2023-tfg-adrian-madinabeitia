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
LINEAR_UMBRALS = [3.5, 5.0, float('inf')]
ANGULAR_UMBRALS = [-0.35, -0.10, 0, 0.10, 0.35, float('inf')]
ALTITUDE_UMBRALS = [-1, -0.5, 0, 0.5, 1, float("inf")]

ALL_UMBRALS = [LINEAR_UMBRALS, ANGULAR_UMBRALS, ALTITUDE_UMBRALS]

# General aspects
USE_WEIGHTS = True
USE_AUGMENTATION = False

#Vels dict
velsDict = {"x": 0, "y" : 1, "z" : 2}

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
            resized_image = image.resize((int(224), int(224)))

            image_array = np.array(resized_image)
            images.append(image_array)

    except FileNotFoundError:
        print("Error: File not found.")

    return images



def get_vels(folder_path, spaceComponent):
    velsLineal = []
    velsAngular = []
    velsAltitude = []

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

                if len(numbers) >= 3:
                    velsAltitude.append(numbers[2])

                velsLineal.append(abs(numbers[0]))
                velsAngular.append(numbers[1])

    except FileNotFoundError:
        print("Error: File not found.")
    
    allVels = [velsLineal, velsAngular, velsAltitude]
    
    return allVels[velsDict[spaceComponent]]



# spaceComponent = "x", "y" or "z"
class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform, spaceComponent) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.spaceComponent = spaceComponent
        self.minSamples = 2
        
        self.imgData = []
        self.velData = []

        for dir in main_dir:
            self.imgData.extend(get_image_dataset(dir + "/frontal_images"))

            # Gets each the component of the selected velocity
            self.velData.extend(get_vels(dir + "/labels", spaceComponent))

        # Joins velocity's and images
        self.dataset = []
        self.image_shape = self.imgData[0].shape
        for i in range(len(self.velData)):
            self.dataset.append((self.imgData[i], self.velData[i]))
        

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = np.array(self.dataset[index][1])
        data = Image.fromarray(img)

        if self.transform is not None:
            data = self.transform(data)

        return data, label

    def subSample(self, label, percent):
        toDel = int(len(label) * percent)
        subsample = random.sample(label, len(label) - toDel)

        return subsample

    def getVelSubset(self, dataset, lowerBound, upperBound):
        linearSub = [(img, vel) for img, vel in dataset if lowerBound < vel <= upperBound]

        return linearSub
    
    
    def balancedDataset(self):
        useWeights = True

        # Combined dataset
        weights = [0.6, 0.4, 1.0, 1.0, 0.4, 0.6]
                
        # Gets all the subsets
        umbralIndex = velsDict[self.spaceComponent]
        labeledDataset = [self.getVelSubset(self.dataset, float('-inf'), ALL_UMBRALS[umbralIndex][0])]
        
        for i in range(0, len(ALL_UMBRALS[umbralIndex]) - 1):
            labeledDataset.append(self.getVelSubset(self.dataset, ALL_UMBRALS[umbralIndex][i], ALL_UMBRALS[umbralIndex][i+1]))


        # Gets the max number of samples in a label
        maxSamples = 0
        
        for velLabeled in labeledDataset:
            if len(velLabeled) > maxSamples:
                maxSamples = len(velLabeled)     
        
        # Balances all the classes
        balancedDataset = []

        for i, currentSubset in enumerate(labeledDataset):
            balancedAngLabeled = []
            
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
                    balancedSubset = self.subSample(balancedSubset, 1 - weights[i])

                    
                balancedDataset.extend(balancedSubset)

        
        return balancedDataset

