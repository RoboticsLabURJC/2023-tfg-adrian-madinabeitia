from torch.utils.data import DataLoader
import torch
from data import rosbagDataset
from train import DATA_PATH, dataset_transforms, load_checkpoint
from models import pilotNet
import numpy as np


dataset = rosbagDataset(DATA_PATH, dataset_transforms)
train_loader = DataLoader(dataset)

# quit = False


# def on_quit():
#     global quit
#     quit = True

model = pilotNet()
load_checkpoint(model)

device = torch.device("cuda:0")
model.to(device)

for i, data in enumerate(train_loader, 0):
    # if quit:
    #     break

    label, image = data

    # Prints the image pixels
    # print("For", dataset.dataset[i][0])
    print("------------------------------------------")
    print("Expected", dataset.dataset[i][1])

    print("And got", model(image)[0].tolist())