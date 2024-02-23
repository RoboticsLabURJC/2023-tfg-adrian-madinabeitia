#!/usr/bin/env python3

# Execute as => python3 train.py --n network2.tar

from rosbags.rosbag2 import Reader as ROS2Reader
import torch
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import os
from PIL import Image

ANGULAR_UMBRALS = [-0.3, -0.1, 0, 0.1, 0.3, float('inf')]  # label < umbral
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
    def __init__(self, main_dir, transform, boolAug=False, dataAument=1) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.minSamples = 25
        self.dataAument = dataAument
        
        self.imgData = get_image_dataset(main_dir + "/frontal_images")
        self.velData = get_vels(main_dir + "/labels")

        self.dataset =  self.get_dataset() 
        self.applyAug = boolAug
        

    def __len__(self):
        return len(self.dataset)

    def get_dataset(self):
        dataset = list(((self.imgData[i], self.velData[i]) for i in range(len(self.velData))))

        original_size = len(dataset)
        new_size = int(original_size * self.dataAument)

        for _ in range(new_size - original_size):
            randIndex = random.randint(0, original_size - 1)
            dataset.append((self.imgData[randIndex], self.velData[randIndex]))
        
        return dataset



    def __getitem__(self, item):
        device = torch.device("cuda:0")
        
        # Applys augmentation
        if self.applyAug:
            image_tensor = torch.tensor(self.dataset[item][0]).to(device)
            image_tensor, vel_tensor = self.applayAugmentation(device, image_tensor, self.dataset[item][1])

        # Not applys augmentation
        else:
            image_tensor = self.transform(self.dataset[item][0]).to(device)
            vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)
    
    
    def applayAugmentation(self, device, imageTensor, velocityValue):
        mov = 10

        # Aplys color augmentation 
        augmented_image_tensor = self.colorAugmeentation()(image=imageTensor.cpu().numpy())['image']
        imgTensor = torch.tensor(augmented_image_tensor).to(device)

        randNum = random.randint(0, 7)

        # Flips the image so the angular will be in opposite way    
        if randNum == 3 or randNum == 4:    # 2 / 8 chance
            imgTensor = torch.flip(imgTensor, dims=[-1])
            velTensor = torch.tensor((velocityValue[0], -velocityValue[1])).to(device)

        # Horizontal movement
        elif randNum == 2:                  # 1 / 8 chance
            randMove = random.randint(-mov, mov)
            imgTensor = F.affine(imgTensor, angle=0, translate=(randNum, 0), scale=1, shear=0)
            velTensor = torch.tensor((velocityValue[0], velocityValue[1] + randMove/(mov*2))).to(device)

        # Vertical movement
        elif randNum == 1:                  # 1 / 8 chance
            randMove = random.randint(-mov, mov)
            imgTensor = F.affine(imgTensor, angle=0, translate=(0, randMove), scale=1, shear=0)
            velTensor = torch.tensor((velocityValue[0] - randMove/(mov*1.5), velocityValue[1])).to(device)
        
        else:   # No changes                # 4 / 8 chance
            velTensor = torch.tensor(velocityValue).to(device)

        return imgTensor, velTensor

    def colorAugmeentation(self, p=0.5):
        return A.Compose([
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(),
            A.Normalize(),
            ToTensorV2(),
        ])
        

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
        useWeights = False
        #   <       5     5.5   inf
        weights = [(0.80, 0.0, 0.0),     # < -0.6
                   (0.90, 0.7, 0.4),     # < -0.3
                   (0.50, 0.5, 0.99),    # < 0
                   (0.50, 0.5, 0.99),    # < 0.3
                   (0.80, 0.7, 0.4),     # < 0.6
                   (0.90, 0.0, 0.0)]      # < inf


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

        # Auments the data
        maxSamples = int(maxSamples * self.dataAument)        
        
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
                    
                    # Subsamples if it is
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


        return rawDataset


dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])


if __name__ == "__main__":

    # Crear una instancia de tu conjunto de datos
    dataset = rosbagDataset(main_dir=DATA_PATH, transform=dataset_transforms, boolAug=True, dataAument=2)

    # Mostrar algunas imágenes y sus velocidades asociadas
    for i in range(15):  # Mostrar las primeras 5 imágenes
        idx = random.randint(0, len(dataset) - 1)
        velocity, image = dataset[idx]

        # Deshacer el cambio de forma y la normalización para mostrar la imagen correctamente
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 0.5) + 0.5  # Desnormalizar

        # Mostrar la imagen y su velocidad asociada
        plt.imshow(image)
        plt.title(f"Velocidad: {velocity}")
        plt.show()