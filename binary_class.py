import torch
import torch.nn as nn

# Multiclass problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # 0 for negative and 1 for values>0 - most popular choice
        self.linear2 = nn.Linear(hidden_size, 1) # 1 is fixed in this case

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        y_pred = torch.sigmoid(out) # implement sigmoid for binary classes - usually in the last layer of the binary classification problem
        return out
    
model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() # applies softmax by default