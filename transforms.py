'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data during creation of the dataset

On images
-----------
CenterCrop, Grayscale, Pad, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomRotation, Resize, Scale

On Tensors
-----------
LinearTransformation, Normalize, RandomErasing

Conversion
-----------
ToPILImage: from tensor or ndarray
To Tensor: from numpy.ndarray or PILImage

Generic
-----------
Use Lambda

Custom
-----------
Write own class

Compose multiple transforms
--------------
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

torchvision.transforms.Rescale(256)
torchvision.transforms.ToTensor()
'''

from typing import Any
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# own custom dataset

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # dataloading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]# size = n_samples, 1 retains original shape of the numpy array
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        # dataset[0]
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)

        return sample
     
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor 

    def __call__(self, sample) -> Any:
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=None)
first_dataset = dataset[0]
features, labels = first_dataset
print(features)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_dataset = dataset[0]
features, labels = first_dataset
print(features)
print(type(features), type(labels))