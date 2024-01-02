import torch

x = torch.randn(3, requires_grad=True)
print(x)

# y = x + 2
# print(y)
# z = y * y * 2
# z = z.mean()
# print(z)

# z.backward() #dz/dx
# print(x.grad)

# x.require_grad_(False)
# x.detach()
# with toch.no_grad()

# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

# y= x + 2
# print(y)

# with torch.no_grad():
#     y = x + 2
#     print(y)

weights = torch.ones(4, requires_grad=True)

# for epoch in range(2):
#     model_output = (weights*3).sum()

#     model_output.backward()

#     print(weights.grad) # accumulates gradient

#     weights.grad.zero_() # to prevent gradient accumulation
