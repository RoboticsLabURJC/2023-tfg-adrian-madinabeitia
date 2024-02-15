#!/usr/bin/env python3

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.serialization import save
from torch.utils.data import DataLoader

import torch.optim as optim
import sys

import ament_index_python

# Package includes
package_path = ament_index_python.get_package_share_directory("drone_driver")
sys.path.append(package_path)

from include.data import rosbagDataset, dataset_transforms, DATA_PATH
from include.models import pilotNet

writer = SummaryWriter()


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


def train(checkpointPath, model: pilotNet, optimizer: optim.Optimizer):

    # Mean Squared Error Loss
    criterion = nn.MSELoss()

    dataset = rosbagDataset(DATA_PATH, dataset_transforms)
    dataset.balancedDataset()

    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    print("Starting training")
    
    consecutiveEpochs = 1  # Contador para el número de épocas consecutivas con pérdida media <= 0.5
    averageLoss = 0.5
    limitEpochs = 1000

    for epoch in range(limitEpochs):

        epoch_loss = 0.0  # Pérdida total de la época

        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            label, image = data

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

            # Saves the model
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss))
                save_checkpoint(checkpointPath, model, optimizer)

        # Loss each epoch
        averageLoss = epoch_loss / len(train_loader)
        print('Epoch {} - Average Loss: {:.3f}'.format(epoch + 1, averageLoss))

        # End criteria succeded?
        if averageLoss <= averageLoss:
            consecutiveEpochs += 1
        else:
            consecutiveEpochs = 0

        # Ends the training in n consecutive epochs with creteria succeded
        if consecutiveEpochs == consecutiveEpochs:
            print('Training terminated. Consecutive epochs with average loss <= 0.5.')
            break

    print('Finished Training')


if __name__ == "__main__":

    # Gets the model path
    package_path = ament_index_python.get_package_share_directory("drone_driver")
    checkpointPath = package_path + "/utils/network.tar"

    model = pilotNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.05)

    # Loads iif theres another model
    if should_resume():
        load_checkpoint(checkpointPath, model, optimizer)

    # Set up cura
    device = torch.device("cuda:0")
    model.to(device)
    model.train(True)

    # Trains
    train(
        checkpointPath,
        model,
        optimizer,
    )