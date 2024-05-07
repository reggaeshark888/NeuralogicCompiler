import numpy as np

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

def print_network_parameters(network):
    # Print weights and biases
    for name, param in network.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(param.data)

def create_XOR_dataset():
    x1 = np.array([0.,0.,1.,1.], dtype=np.float32)
    x2 = np.array([0.,1.,0.,1.], dtype=np.float32)
    y = np.array([0.,0.,0.,1.], dtype=np.float32)

    x1 = np.repeat(x1, 50)
    x2 = np.repeat(x2, 50)
    y = np.repeat(y, 50)
    
    x1 = x1 + np.random.rand(x1.shape[0])*0.05
    x2 = x2 + np.random.rand(x2.shape[0])*0.05

    #shuffle
    index_shuffle = np.arange(x1.shape[0])
    np.random.shuffle(index_shuffle)

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    y = y.astype(np.float32)

    x1 = x1[index_shuffle]
    x2 = x2[index_shuffle]
    y = y[index_shuffle]

    # convert data to tensors
    x1_torch = torch.from_numpy(x1).clone().view(-1,1)
    x2_torch = torch.from_numpy(x1).clone().view(-1,1)
    y_torch = torch.from_numpy(x1).clone().view(-1,1)

    X = torch.hstack([x1_torch, x2_torch])

    X_train = X[:150, :]
    X_test = X[150:, :]
    y_train = y_torch[:150, :]
    y_test = y_torch[150:, :]

    return X_train, y_train, X_test, y_test

def train(model, loss_function, optimizer, x, y, iter):
    # store loss for each epoch
    all_loss = []

    for epoch in range(iter):
        # forward pass
        y_hat = model(x)

        # calculate the loss
        loss = loss_function(y_hat, y)
        all_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return all_loss

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    

# dataset
X_train, X_test, y_train, y_test = create_XOR_dataset()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

# model_XOR = TwoLayerMLP()

# # define the loss
# loss_function = torch.nn.BCELoss()


# optimizer = torch.optim.SGD(model_XOR.parameters(), lr=0.01)
# all_loss = train(model_XOR, loss_function, optimizer, X_train, y_train, 500000)

