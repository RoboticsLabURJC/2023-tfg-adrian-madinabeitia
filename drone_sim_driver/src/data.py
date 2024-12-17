#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import albumentations as A
from torchvision import transforms
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import os
from PIL import Image

# General velocity's
MAX_ANGULAR = 1.5
MAX_LINEAR = 7.0
MIN_LINEAR = 0.0

# Labels

# Follow line
# ANGULAR_UMBRALS = [-0.5, -0.2, 0, 0.2, 0.5, float('inf')]  # label < umbral
# LINEAR_UMBRALS = [4.5, 5.5, float('inf')]

# Gates (human pilot)
# ANGULAR_UMBRALS = [-0.7, -0.2, 0, 0.2, 0.7, float('inf')]
# LINEAR_UMBRALS = [3.0, 4.25, float('inf')]

# Gates expert pilot
# ANGULAR_UMBRALS = [-0.45, -0.15, 0, 0.15, 0.45, float('inf')]
# LINEAR_UMBRALS = [2.0, 3.25, float('inf')]

# Gates extended
ANGULAR_UMBRALS = [-0.25, -0.10, 0, 0.10, 0.25, float('inf')]
LINEAR_UMBRALS = [2.5, 3.0, float('inf')]

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
    def __init__(self, main_dir, transform, boolAug=USE_AUGMENTATION, dataAugment=1) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.minSamples = 1
        self.dataAugment = dataAugment
        
        self.imgData = []
        self.velData = []
        for dir in main_dir:
            self.imgData.extend(get_image_dataset(dir + "/frontal_images"))
            #self.imgData.extend(get_image_dataset(dir + "/proccesedImages"))
            self.velData.extend(get_vels(dir + "/labels"))

        self.dataset =  self.get_dataset() 
        self.applyAug = boolAug
        

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
        norm = (lineal, angular)

        # Apply augmentation
        if self.applyAug:
            image_tensor = torch.tensor(self.dataset[item][0]).to(device)
            image_tensor, vel_tensor = self.applyAugmentation(device, image_tensor, norm)

        # Not apply augmentation
        else:
            #! cHANGE UINT8 TO device
            image_tensor = self.transform(self.dataset[item][0]).to(torch.float32)
            vel_tensor = torch.tensor(norm).to(device)

            
        return (vel_tensor, image_tensor)    
    
    def applyAugmentation(self, device, imgTensor, velocityValue):
        mov = 2

        # Apply color augmentation 
        augmented_image_tensor = self.colorAugmentation()(image=imgTensor.cpu().numpy())['image']
        imageTensor = torch.tensor(augmented_image_tensor).clone().detach().to(device)


        randNum = random.randint(0, 7)

        # Flips the image so the angular will be in opposite way    
        if randNum == 3 or randNum == 4:    # 2 / 8 chance
            imageTensor = torch.flip(imageTensor, dims=[-1])
            velTensor = torch.tensor((velocityValue[0], -velocityValue[1])).to(device)

        # Horizontal movement
        elif randNum == 20 or randNum == 10:                  # 1 / 8 chance
            randMove = random.randint(-mov, mov)
            imageTensor = F.affine(imageTensor, angle=0, translate=(randNum, 0), scale=1, shear=0)
            velTensor = torch.tensor((velocityValue[0], velocityValue[1] + randMove/(mov*4))).to(device)


        else:   # No changes                # 4 / 8 chance
            velTensor = torch.tensor(velocityValue).to(device)

        finalImgTensor = self.transform(imageTensor.cpu().numpy()).to(torch.float32)
        # finalImgTensor = self.transform(imageTensor.cpu().numpy()).to(torch.float32)
        return finalImgTensor, velTensor

    def colorAugmentation(self, p=0.5):
        return A.Compose([
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(),
            A.Normalize()
        ])
        

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
        #   <       5     5.5   inf       Weights 1
        # weights = [(0.85, 0.15, 0.25),     # < -0.6
        #            (0.65, 0.55, 0.45),     # < -0.3
        #            (0.35, 0.75, 0.995),    # < 0
        #            (0.35, 0.75, 0.995),    # < 0.3
        #            (0.65, 0.55, 0.45),     # < 0.6
        #            (0.85, 0.15, 0.25)]      # < inf

        weights = [(0.65, 0.35, 0.25),     # < -0.6
                   (0.45, 0.55, 0.45),     # < -0.3
                   (0.25, 0.75, 0.995),    # < 0
                   (0.25, 0.75, 0.995),    # < 0.3
                   (0.45, 0.55, 0.45),     # < 0.6
                   (0.65, 0.35, 0.25)]      # < inf
        # Gates => Remote pilot
        # weights = [(0.2, 0.1, 0.0),
        #         (0.55, 0.65, 0.25), 
        #         (0.95, 0.95, 0.75), 
        #         (0.95, 0.95, 0.75), 
        #         (0.55, 0.65, 0.25),  
        #         (0.2, 0.1, 0.0)] 

        # Combined dataset
        # weights = [(0.7, 0.65, 0.5),     # < -0.6
        #            (0.75, 0.85, 0.75),     # < -0.3
        #            (0.70, 0.90, 0.80),    # < 0
        #            (0.70, 0.90, 0.80),    # < 0.3
        #            (0.75, 0.85, 0.75),     # < 0.6
        #            (0.7, 0.65, 0.5)]      # < inf

        # All vars active
        # weights = [(1.0, 1.00, 1.0),     # < -0.6
        #            (1.0, 1.00, 1.0),     # < -0.3
        #            (1.0, 1.00, 1.0),    # < 0
        #            (1.0, 1.00, 1.0),    # < 0.3
        #            (1.0, 1.00, 1.0),     # < 0.6
        #            (1.0, 1.00, 1.0)]      # < inf
                

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

