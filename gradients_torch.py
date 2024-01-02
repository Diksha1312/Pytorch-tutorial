"""
************STEP 2********************
1. Prediction: manual
2. Gradients computation: autograd
3. Loss computation: manual
4. Param updates: manual

"""
import torch
# f = w*x

# f = 2*x

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# # gradient
# # MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N 2x (w*x - y)

# def gradient(x,y,y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training

learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    # gradients
    #dw = gradient(x,y,y_pred)
    l.backward() # calculates grad of loss wrt w dl/dw

    # update weights
    #w -= learning_rate * dw
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')