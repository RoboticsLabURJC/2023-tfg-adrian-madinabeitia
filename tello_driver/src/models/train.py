#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.serialization import save
from torch.utils.data import DataLoader
import argparse
import torch.optim as optim
import sys

import ament_index_python

# Package includes
package_path = ament_index_python.get_package_share_directory("tello_driver")
sys.path.append(package_path)

from src.dataset.data import rosbagDataset
import src.dataset.transforms as transforms

from models import pilotNet

writer = SummaryWriter()



BATCH_SIZE = 100
LEARNING_RATE=1e-4
MOMENT=0.05

TARGET_LOSS = 0.005#0.05
TARGET_CONSECUTIVE_LOSS = 4

def should_resume():
    return "--resume" in sys.argv or "-r" in sys.argv


def save_checkpoint(path, model: pilotNet, optimizer: optim.Optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model: pilotNet, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train(checkpointPath, datasetPath, model: pilotNet, optimizer: optim.Optimizer):

    # Mean Squared Error Loss
    criterion = nn.MSELoss()


    # Use pilotNet_transforms or deepPilot_Transforms(aug)
    dataset = rosbagDataset(datasetPath, transform=transforms.deepPilot_Transforms(None))
    dataset.balancedDataset()

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Starting training")
    
    consecutiveEpochs = 0  
    targetLoss = TARGET_LOSS
    epoch = 0

    while consecutiveEpochs != TARGET_CONSECUTIVE_LOSS:

        epoch_loss = 0.0  
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            label, image = data

            #* This is for deepPilot
            label = [x[0] for x in label]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, label)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Loss graphic
            writer.add_scalar("Loss/train", loss, epoch)

            # Saves the model each 20 iterations
            if i % 20 == 0:
                save_checkpoint(checkpointPath, model, optimizer)

        epoch += 1

        # Loss each epoch
        averageLoss = epoch_loss / len(train_loader)
        print('Epoch {} - Average Loss: {:.3f}'.format(epoch, averageLoss))

        # End criteria succeeded?
        if averageLoss <= targetLoss:
            consecutiveEpochs += 1
        else:
            consecutiveEpochs = 0

    print('Training terminated. Consecutive epochs with average loss <= ', targetLoss)


def main(filePath, datasetPath):

    model = pilotNet()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENT)

    # Loads if theres another model
    if should_resume():
        load_checkpoint(filePath, model, optimizer)

    # Set up cura
    device = torch.device("cuda:0")
    model.to(device)

    model.train(True)
    train(filePath, datasetPath,
        model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ROS bags and plot results.')

    parser.add_argument('--network_path', type=str, default="../utils/net1")
    parser.add_argument('--dataset_path', type=str, default="../training_dataset/network")

    args = parser.parse_args()
    main(args.network_path, args.dataset_path)