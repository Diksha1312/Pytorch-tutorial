import torch
import torchvision
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('Softmax numpy:', outputs)

'''softmax will squash the inputs to be outputs between 0 and 1 so that we have a probability as an output 
- good choice in the last layer of a multiclass classification problem
'''
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim = 0)
print(outputs)