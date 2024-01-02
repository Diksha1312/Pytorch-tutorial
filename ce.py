import torch
import torch.nn as nn
import numpy as np

# def cross_entropy(actual, predicted):
#     loss = -np.sum(actual * np.log(predicted))
#     return loss # / float(predicted.shpe[0])


# # y must be one hot encoded
# # if class 0 : [1, 0, 0]
# # if class 1 : [0, 1, 0] and so on
# Y = np.array([1,0,0])
# # y_pred has probablities
# y_pred_good = np.array([0.7, 0.2, 0.1])
# y_pred_bad = np.array([0.1, 0.3, 0.6])
# l1 = cross_entropy(Y, y_pred_good)
# l2 = cross_entropy(Y, y_pred_bad)
# print(f"Good {l1:.4f}")
# print(f"Bad {l2:.4f}")

loss = nn.CrossEntropyLoss()
# 3 samples
Y = torch.tensor([2, 0, 1])
# nsamples x nclasses = 1x3
# new size = 3x3
Y_prediction_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.0, 3.0, 0.1]])
Y_prediction_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_prediction_good, Y)
l2 = loss(Y_prediction_bad, Y)
print(f"Good {l1.item():.4f}")
print(f"Bad {l2.item():.4f}")

_, predictions1 = torch.max(Y_prediction_good, 1)
_, predictions2 = torch.max(Y_prediction_bad, 1)

print(predictions1, predictions2)
