"""
************STEP 3********************
1. Prediction: manual
2. Gradients computation: autograd
3. Loss computation: pytorch loss
4. Param updates: pytorch optimizer

************STEP 4********************
1. Prediction: pytorch model
2. Gradients computation: autograd
3. Loss computation: pytorch loss
4. Param updates: pytorch optimizer

************PIPELINE*******************
1. Design model (input, output, size, forward pass)
2. Construct loss and optimizer
3. Training loop
-------forward pass: compute prediction
-------backward pass: gradients
-------update weights

"""

import torch
import torch.nn as nn
# f = w*x

# f = 2*x

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
print(x.shape)
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
        
model = LinearRegression(input_size, output_size)

print(f"Prediction before training: f(5) = {model(x_test).item():.3f}")

# Training

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    # gradients
    #dw = gradient(x,y,y_pred)
    l.backward() # calculates grad of loss wrt w dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters() # unpack them - list of list
        print(f'epoch {epoch+1}: w = {w[0][0]:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')