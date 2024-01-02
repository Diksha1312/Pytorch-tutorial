'''
epoch = 1 forward and backward pass of all training samples
batch_size = number of training samples in one forward and backward pass
number of iterations = number of passes, each pass using [batch_size] number of samples

ex. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# own custom dataset

class WineDataset(Dataset):

    def __init__(self):
        # dataloading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # size = n_samples, 1 retains original shape of the numpy array
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

for batch in dataloader:
    features, labels = batch
    print(features, labels)
    break

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)
