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

def print_network_parameters_for_neurons(network):
    # Print weights and biases for each neuron
    for name, param in network.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            if name.endswith('weight'):  # If the parameter is a weight
                for idx, weights in enumerate(param):
                    print(f"  Neuron {idx + 1} weights: {weights.data.numpy()}")
            elif name.endswith('bias'):  # If the parameter is a bias
                for idx, bias in enumerate(param):
                    print(f"  Neuron {idx + 1} bias: {bias.data.item()}")

def graph_neural_network(network):
    graph = make_dot(params=dict(list(network.named_parameters())))
    graph.render('network_graph', format='png', cleanup=True)  # This saves and cleans up the dot file

def create_torch_XOR_dataset():
    x1 = np.array ([0., 0., 1., 1.], dtype = np.float64)
    x2 = np.array ([0., 1., 0., 1.], dtype = np.float64)
    y  = np.array ([0., 1., 1., 0.],dtype = np.float64)

    x1 = np.repeat(x1, 50)
    x2 = np.repeat(x2, 50)  
    y =  np.repeat(y,  50)
    
    # Add noise
    x1 = x1 + np.random.rand(x1.shape[0])*0.05
    x2 = x2 + np.random.rand(x2.shape[0])*0.05

    # Shuffle the data
    index_shuffle = np.arange(x1.shape[0])
    np.random.shuffle(index_shuffle)

    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    y = y.astype(np.float64)

    x1 = x1[index_shuffle]
    x2 = x2[index_shuffle]
    y = y[index_shuffle]

    # Convert data to pytorch tensors
    x1_torch = torch.from_numpy(x1).clone().view(-1,1)
    x2_torch = torch.from_numpy(x2).clone().view(-1,1)
    y_torch = torch.from_numpy(y).clone().view(-1,1)

    X = torch.hstack([x1_torch, x2_torch])

    X_train = X[:150,:]
    X_test  = X[150:,:]
    y_train = y_torch[:150,:]
    y_test  = y_torch[150:,:]

    return X_train, y_train, X_test, y_test

def train(model, loss_function, optimizer, x, y, no_of_epochs):
    # store loss for each epoch
    all_loss = []

    for epoch in range(no_of_epochs):
        # forward pass
        y_hat = model(x)

        # calculate the loss
        loss = loss_function(y_hat, y)
        all_loss.append(loss.item())
        
        # clears out the old gradients from the previous step
        optimizer.zero_grad()
        loss.backward()
        # takes a step in the parameter step opposite to the gradient, peforms the update rule
        optimizer.step()

        print(all_loss[epoch])

    return all_loss

def create_masks(param_tensor, threshold=0.5):
    greater_than_threshold = param_tensor >= threshold
    less_than_threshold = param_tensor <= -threshold
    return greater_than_threshold, less_than_threshold

def create_center_mask(param_tensor, threshold=1):
    return (param_tensor > -threshold) & (param_tensor < threshold)

def calculate_penalty(tensor, above_threshold, LAMBDA):
    if above_threshold:
        return (tensor - 10) * 2 * LAMBDA
    else:
        return (tensor + 15) * 2 * LAMBDA

def calculate_central_penalty(tensor, LAMBDA):
    return tensor * 2 * LAMBDA

def calculate_integer_penalty(model, lambda_integer=0.05):
    total_penalty = 0
    for _, param in model.named_parameters():
        # calculate the update based on the mask and update parameters
        gt_penalty = calculate_penalty(param.data, True, lambda_integer)
        ls_penalty = calculate_penalty(param.data, False, lambda_integer)

        gt_mask, ls_mask = create_masks(param.data)

        total_penalty += torch.sum(gt_penalty*gt_mask) + torch.sum(ls_penalty*ls_mask)

    return total_penalty

def calculate_total_opposite_sign_penalty(model, LAMBDA_POLARITY=0.025):
    total_penalty = 0
    epsilon = 1e-8  # Small constant
    # iterate over all layers of the model
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            bias = layer.bias
            # Compute the penalty for this layer's weights and bias
            penalty = torch.sum(LAMBDA_POLARITY * torch.max(torch.zeros_like(weights), weights * torch.sign(bias).unsqueeze(1)))
                    # TODO: add the loss 
                    #+ torch.sum(lambda_sign * torch.max(torch.zeros_like(weights), weights * torch.sign(bias).unsqueeze(1)))
            total_penalty += penalty
    
    return total_penalty + epsilon

def calculate_integer_penalty2(model, LAMBDA_INTEGERS=0.01):
    total_penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:
            # Finding the nearest multiples for 10 and -15
            # We use the floor division // to find the closest integer factor and then multiply back
            nearest_10 = torch.round(param / 10.0) * 10
            nearest_minus_15 = torch.round(param / -15.0) * -15
            
            # Calculate squared differences from the nearest multiples
            penalty_10 = (param - nearest_10).pow(2)
            penalty_minus_15 = (param - nearest_minus_15).pow(2)
            
            # Take the minimum of the squared differences for each element
            element_penalty = torch.min(penalty_10, penalty_minus_15)
            
            # Sum up all penalties
            total_penalty += element_penalty.sum()

    # Scale the penalty by the regularization strength lambda
    return LAMBDA_INTEGERS * total_penalty

def calculate_integer_penalty3(model, targets = [10, -10, -15, 15], LAMBDA_INTEGERS = 0.01):
    """
    Calculate the regularization penalty for the model parameters.
    
    Args:
    model (nn.Module): The neural network model.
    targets (list): List of target values for the parameters.
    lambda_reg (float): Regularization strength.

    Returns:
    torch.Tensor: The regularization penalty.
    """
    total_penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:
            # Calculate the minimum squared difference for each parameter
            penalties = [(param - target)**2 for target in targets]
            min_penalty = torch.min(torch.stack(penalties), dim=0)[0]
            # Apply square root to the minimum squared difference
            sqrt_penalty = torch.sqrt(min_penalty)
            # Sum up the penalties
            total_penalty += sqrt_penalty.sum()
    
    return LAMBDA_INTEGERS * total_penalty

def calculate_weight_magnitude_penalty(model, lambda_magnitude = 0.01):
    # TODO: modify to compute the variance of only top two weights
    regularization_loss = 0.0
    for layer in model.children():
        if hasattr(layer, 'weight'):  # Ensure the layer has weights
            # Calculate the mean of weights for each neuron
            weights = layer.weight
            mean_weights = torch.mean(weights, dim=1, keepdim=True)
            # Calculate the mean squared deviation from the mean for each weight
            variance = torch.mean((weights - mean_weights) ** 2, dim=1)
            # Sum up the variances for all neurons in the layer
            total_variance = torch.sum(variance)
            regularization_loss += total_variance
    total_penalty = lambda_magnitude * regularization_loss

    return total_penalty

def train_with_rectified_L2(model, loss_function, optimizer, x, y, 
                            no_of_epochs=90000, ALPHA=0.5,
                            LAMBDA_MAGNITUDE = 0.01,
                            LAMBDA_POLARITY = 0.01,
                            LAMBDA_INTEGERS = 0.01,
                            initial_lr=0.01,
                            max_lr=1):
    # store loss and penalization for each epoch
    all_loss_without_reg = []
    all_loss_with_reg = []

    for epoch in range(no_of_epochs):
        # #Calculate the new learning rate
        # lr = initial_lr + (max_lr - initial_lr) * (epoch / no_of_epochs)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        epsilon = 1e-8  # Small constant to prevent log(0)
        # forward pass
        y_hat_raw = model(x)
        y_hat = torch.clamp(y_hat_raw, epsilon, 1 - epsilon)  # Clamping probabilities

        # compute gradient G1 - loss
        loss = loss_function(y_hat, y)
        

        all_loss_without_reg.append(loss.item())

        # compute opposite sign penalty
        polarity_penalty = calculate_total_opposite_sign_penalty(model, LAMBDA_POLARITY)

        # compute same weight magnitude penalty
        magnitude_penalty = calculate_weight_magnitude_penalty(model, LAMBDA_MAGNITUDE)

        # compute distance from integer numbers penalty
        # integer_penalty = calculate_integer_penalty2(model_XOR)
        # integer_penalty = calculate_integer_penalty(model_XOR)
        integer_penalty = calculate_integer_penalty3(model, LAMBDA_INTEGERS=LAMBDA_INTEGERS)

        # total loss with regularization
        if epoch > 1000:
            total_loss = loss + polarity_penalty + magnitude_penalty + integer_penalty
        else:
            total_loss = loss + polarity_penalty + magnitude_penalty

        if total_loss < 0.02:
            print("Optimal solution found")
            return

        all_loss_with_reg.append(total_loss.item())

        # clears out the old gradients from the previous step
        optimizer.zero_grad()
        # parameter update according to G1
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # takes a step in the parameter step opposite to the gradient, peforms the update rule
        optimizer.step()


        print("epoch", epoch)
        print("loss without reg", all_loss_without_reg[epoch])
        print("loss with reg", all_loss_with_reg[epoch])
        #print("learning rate", lr)
        print_network_parameters_for_neurons(model)

        z = 0

    return all_loss_without_reg, all_loss_with_reg
        

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super(TwoLayerMLP, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)
    
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    
class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super(ThreeLayerMLP, self).__init__()
        # Define the first layer with 4 input features and 4 output neurons
        self.layer1 = nn.Linear(4, 4)
        # Define the second layer with 4 input neurons (from layer1) and 2 output neurons
        self.layer2 = nn.Linear(4, 2)
        # Define the output layer with 2 input neurons (from layer2) and 1 output neuron
        self.output_layer = nn.Linear(2, 1)
    
    def forward(self, x):
        # Forward pass through the first layer followed by a sigmoid activation
        x = torch.sigmoid(self.layer1(x))
        # Forward pass through the second layer followed by a sigmoid activation
        x = torch.sigmoid(self.layer2(x))
        # Forward pass through the output layer followed by a sigmoid activation
        x = torch.sigmoid(self.output_layer(x))
        return x




# example usage
if __name__ == '__main__':
    # HYPERPARAMETERS
    INIITIAL_LEARNING_RATE  = 0.001  # Starting learning rate
    MAXIMUM_LEARNING_RATE = 0.5   # Maximum learning rate
    EPOCHS = 30000
    ALPHA = 0.5
    LAMBDA_MAGNITUDE = 0.1
    LAMBDA_POLARITY = 0.01
    LAMBDA_INTEGERS = 0.005

    # dataset
    X_train, y_train, X_test, y_test = create_torch_XOR_dataset()

    model_XOR = TwoLayerMLP().double()
    # define the loss
    loss_function = torch.nn.BCELoss()
    #optimizer = torch.optim.SGD(model_XOR.parameters(), lr=INIITIAL_LEARNING_RATE)
    optimizer = torch.optim.Adam(model_XOR.parameters(), lr=INIITIAL_LEARNING_RATE, eps=1e-7)

    all_loss = train_with_rectified_L2(model_XOR, 
                                    loss_function, 
                                    optimizer, 
                                    X_train, 
                                    y_train,
                                    no_of_epochs=EPOCHS,
                                    ALPHA=ALPHA,
                                    LAMBDA_MAGNITUDE=LAMBDA_MAGNITUDE,
                                    LAMBDA_POLARITY=LAMBDA_POLARITY,
                                    LAMBDA_INTEGERS=LAMBDA_INTEGERS,
                                    initial_lr=INIITIAL_LEARNING_RATE,
                                    max_lr=MAXIMUM_LEARNING_RATE)


# TODO: move into a function
# # parameter update according to G2
# if epoch > 10000:
#     for _, param in model.named_parameters():
#         # apply the masking to determine whether parameter is above or below threshold
#         gt_mask, ls_mask = create_masks(param.data, ALPHA)

#         # calculate the update based on the mask and update parameters
#         gt_penalty = calculate_penalty(param.data, True, LAMBDA)
#         ls_penalty = calculate_penalty(param.data, False, LAMBDA)

#         # Apply the updates
#         param.data = torch.where(
#             gt_mask,
#             param.data - gt_penalty,
#             param.data
#         )
        
#         param.data = torch.where(
#             ls_mask,
#             param.data - ls_penalty,
#             param.data
#         )

#         # apply the J=0 case in the last few iterations
#         if no_of_epochs - epoch < 5:
#             ct_mask = create_center_mask(param.data, ALPHA)
#             ct_penalty = calculate_central_penalty(param.data, LAMBDA)

#             param.data = torch.where(
#                 ct_mask,
#                 ct_penalty,
#                 param.data
#             )