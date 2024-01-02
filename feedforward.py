# MNIST
# Dataloader, transformations
# multilayer neural network, activation function
# loss and optimizer
# training loop (batch training)
# model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper params
input_size = 784 # 28 * 28
hidden_size = 100
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./pytorch_tutorial/data/mnist', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root='./pytorch_tutorial/data/mnist', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

for features, labels in train_loader:
    print(features.shape, labels.shape)
    break

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(features[i][0], cmap='gray')
#plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)
model.to(device)

#print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1,784) # [100,1,28,28] to [100,784]
        images = images.to(device)
        labels = labels.to(device)
        # forward
        result = model(images)
        loss = criterion(result, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch: {epoch+1} / {num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1,784) # [100,1,28,28] to [100,784]
        images = images.to(device)
        labels = labels.to(device)
        # forward
        result = model(images)

        _, predictions = torch.max(result,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy = {acc}')

        



