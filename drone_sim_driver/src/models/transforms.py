import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np


# Without augmentation 

pilotNet_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])


# from torchvision import transforms
# import torchvision.transforms.functional as F
# import torch

# class ToUint8(object):
#     def __call__(self, img):
#         return (img * 255).to(torch.uint8)

# class PosterizeTensor(object):
#     def __init__(self, bits):
#         self.bits = bits

#     def __call__(self, img):
#         return F.posterize(img, self.bits)

# # Ensure the transformation pipeline is correctly ordered
# pilotNet_transforms = transforms.Compose([
#     transforms.ToTensor(),  # Convert PIL Image to tensor
#     transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
#     transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1)),
#     ToUint8(),  # Convert to uint8
#     PosterizeTensor(bits=2),  # Apply posterization
#     transforms.Resize([66, 200]),  # Resize the image
# ])

# all_augs_dict = {
#     'gaussian': transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
#     'jitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     'perspective': transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
#     'affine': transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1)),
#     'posterize': transforms.RandomPosterize(bits=2)
# }


# def deepPilot_Transforms(augmentations):

#     augs_to_compose = []

#     if 'auto' in augmentations:
#         for data_aug in all_augs_dict.keys():
#             if np.random.rand() > 0.5:
#                 action = all_augs_dict[data_aug]
#                 augs_to_compose.append(action)
#     elif 'all' in augmentations:
#         for data_aug in all_augs_dict.keys():
#             action = all_augs_dict[data_aug]
#             augs_to_compose.append(action)
#     else:
#         for data_aug in augmentations:
#             action = all_augs_dict[data_aug]
#             augs_to_compose.append(action)

#     augs_to_compose.append(transforms.ToTensor())
#     augs_to_compose.append(transforms.Resize([224, 224]))
#     createdTransform = transforms.Compose(augs_to_compose)

#     return createdTransform