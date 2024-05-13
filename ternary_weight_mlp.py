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
    """
    Generate a dataset for the XOR problem, suitable for training in PyTorch.

    This function creates a dataset representing the XOR logic gate, extends it by repeating
    each data point 50 times, introduces slight random noise to the inputs, and finally shuffles
    the dataset. The dataset is then split into training and testing subsets and converted to
    PyTorch tensors.

    Returns
    -------
    X_train : torch.Tensor
        The training set input features, shape (150, 2). Each row corresponds to an input pair.
    y_train : torch.Tensor
        The training set labels, shape (150, 1). Each entry is the XOR result of the corresponding input pair.
    X_test : torch.Tensor
        The testing set input features, shape (50, 2). Each row corresponds to an input pair.
    y_test : torch.Tensor
        The testing set labels, shape (50, 1). Each entry is the XOR result of the corresponding input pair.

    Notes
    -----
    The dataset initially consists of four points: (0,0), (0,1), (1,0), and (1,1), with corresponding
    XOR outputs: 0, 1, 1, and 0 respectively. Each point is repeated 50 times to increase the dataset size,
    then random noise up to 0.05 is added to the inputs. The entire dataset is shuffled to ensure
    randomness in training and testing splits. The final dataset is divided into 75% training and 25% testing.

    Example
    -------
    >>> X_train, y_train, X_test, y_test = create_torch_XOR_dataset()
    >>> print(X_train.shape, y_train.shape)
    torch.Size([150, 2]), torch.Size([150, 1])
    """
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

def create_torch_XOR_XNOR_dataset():
    """
    Generate a dataset for the combined boolean function (a XOR b) AND (c XNOR d),
    suitable for training in PyTorch.

    This function creates a dataset where each input pair (a, b) and (c, d) is processed
    through XOR and XNOR operations respectively. The final output is true only if
    (a XOR b) AND (c XNOR d) are true. The dataset is replicated, noise is added,
    and then shuffled before splitting into training and testing subsets and converting
    to PyTorch tensors.

    Returns
    -------
    X_train : torch.Tensor
        The training set input features, shape (300, 4). Each row corresponds to four inputs (a, b, c, d).
    y_train : torch.Tensor
        The training set labels, shape (300, 1). Each entry is the result of (a XOR b) AND (c XNOR d).
    X_test : torch.Tensor
        The testing set input features, shape (100, 4). Each row corresponds to four inputs (a, b, c, d).
    y_test : torch.Tensor
        The testing set labels, shape (100, 1). Each entry is the result of (a XOR b) AND (c XNOR d).

    Notes
    -----
    The initial dataset consists of all combinations of binary inputs for (a, b, c, d). Random noise
    up to 0.05 is added to the inputs after replicating each initial combination 25 times. The dataset
    is then shuffled to ensure randomness in training and testing splits. The final dataset is divided
    into 75% training and 25% testing.

    Example
    -------
    >>> X_train, y_train, X_test, y_test = create_torch_XOR_XNOR_dataset()
    >>> print(X_train.shape, y_train.shape)
    torch.Size([300, 4]), torch.Size([300, 1])
    """
    import numpy as np
    import torch

    # Define all combinations of (a, b, c, d)
    abcd = np.array([[a, b, c, d] for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)])
    a, b, c, d = abcd[:, 0], abcd[:, 1], abcd[:, 2], abcd[:, 3]

    # Apply XOR to (a, b) and XNOR to (c, d)
    xor = np.logical_xor(a, b)
    xnor = np.logical_not(np.logical_xor(c, d))
    y = np.logical_and(xor, xnor)

    # Replicate and add noise
    inputs = np.repeat(abcd, 25, axis=0)
    inputs = inputs + np.random.rand(*inputs.shape) * 0.05
    outputs = np.repeat(y, 25)

    # Shuffle the dataset
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    inputs = inputs[indices]
    outputs = outputs[indices]

    # Convert to PyTorch tensors and reshape
    inputs_torch = torch.from_numpy(inputs).double()
    outputs_torch = torch.from_numpy(outputs).double().view(-1, 1)

    # Combine features and split the dataset
    split_idx = int(0.75 * len(inputs_torch))
    X_train, X_test = inputs_torch[:split_idx, :], inputs_torch[split_idx:, :]
    y_train, y_test = outputs_torch[:split_idx, :], outputs_torch[split_idx:, :]

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
    """
    Calculates a regularization penalty which encourages the two largest weights (by magnitude) of each neuron
    in the model's layers to have opposite signs from their corresponding bias.

    Parameters:
    - model (torch.nn.Module): The neural network model.
    - LAMBDA_POLARITY (float, optional): Regularization coefficient. Default is 0.025.

    Returns:
    - total_penalty (float): The computed regularization penalty.
    """
    total_penalty = 0
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
    
    return total_penalty

def calculate_integer_penalty2(model, targets = [10, -10, -15, 15], LAMBDA_INTEGERS=0.01):
    """
    Calculate a regularization penalty that encourages weights and biases of each neuron
    to converge towards specified target values under conditions that weights and biases
    should have opposite polarities, and one should target closer to 15 or -15, while the other
    should target closer to 10 or -10, depending on their relative magnitudes.
    
    Args:
        model (nn.Module): The neural network model.
        LAMBDA_INTEGERS (float): Regularization strength.

    Returns:
        torch.Tensor: The regularization penalty.
    """
    total_penalty = 0.0
    for layer in model.children():
        if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            weights = layer.weight
            biases = layer.bias

            for i in range(weights.shape[0]):  # Iterate over each neuron
                neuron_weights = weights[i]
                neuron_bias = biases[i]

                # Determine targets based on the magnitude relationship of weights and bias
                weight_mean = neuron_weights.mean().item()
                bias_value = neuron_bias.item()

                # Select targets for weights
                if weight_mean > 0:
                    if abs(weight_mean) > abs(bias_value):
                        weight_targets = [15]
                        bias_targets = [-10]
                    else:
                        weight_targets = [10]
                        bias_targets = [-15]
                else:
                    if abs(weight_mean) > abs(bias_value):
                        weight_targets = [-15]
                        bias_targets = [10]
                    else:
                        weight_targets = [-10]
                        bias_targets = [15]

                # Calculate penalties for weights targeting
                weight_penalties = [(neuron_weights - target)**2 for target in weight_targets]
                min_weight_penalty = torch.min(torch.stack(weight_penalties), dim=0)[0]
                sqrt_weight_penalty = torch.sqrt(min_weight_penalty).sum()

                # Calculate penalties for bias targeting
                bias_penalties = [(neuron_bias - target)**2 for target in bias_targets]
                min_bias_penalty = torch.min(torch.stack(bias_penalties), dim=0)[0]
                sqrt_bias_penalty = torch.sqrt(min_bias_penalty)

                # Sum up the penalties for the neuron
                total_penalty += sqrt_weight_penalty + sqrt_bias_penalty

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
    """
    Calculates a regularization penalty which encourages the two largest (by magnitude) weights
    of each neuron in the model's layers to have similar magnitudes.

    Parameters:
    - model (torch.nn.Module): The neural network model.
    - lambda_magnitude (float, optional): Regularization coefficient. Default is 0.01.

    Returns:
    - total_penalty (float): The computed regularization penalty.
    """
    regularization_loss = 0.0

    for layer in model.children():
        if hasattr(layer, 'weight'):  # Ensure the layer has weights
            weights = layer.weight.data  # Use the weight data tensor
            # Obtain the absolute values of weights and sort each row
            sorted_weights, _ = torch.sort(torch.abs(weights), descending=True)
            # Select the top two largest weights by magnitude for each neuron
            top_two_weights = sorted_weights[:, :2]
            # Calculate the mean squared deviation from the mean for the two largest weights
            mean_top_two = torch.mean(top_two_weights, dim=1, keepdim=True)
            variance_top_two = torch.mean((top_two_weights - mean_top_two) ** 2, dim=1)
            # Sum up the variances for all neurons in the layer
            total_variance = torch.sum(variance_top_two)
            regularization_loss += total_variance

    total_penalty = lambda_magnitude * regularization_loss
    return total_penalty

def calculate_sparse_penalty(model, lambda_sparsity=0.01):
    """
    Calculates a regularization penalty which encourages sparsity by penalizing all but the two largest (by magnitude) 
    weights of each neuron in the model's layers.

    Parameters:
    - model (torch.nn.Module): The neural network model.
    - lambda_sparsity (float, optional): Regularization coefficient. Default is 0.01.

    Returns:
    - total_penalty (float): The computed regularization penalty.
    """
    regularization_loss = 0.0

    for layer in model.children():
        if hasattr(layer, 'weight'):  # Ensure the layer has weights
            weights = layer.weight.data  # Use the weight data tensor
            # Obtain the absolute values of weights and sort each row
            sorted_weights, _ = torch.sort(torch.abs(weights), descending=True)
            # Select all weights except the top two largest weights by magnitude for each neuron
            remaining_weights = sorted_weights[:, 2:]  # Start from the third largest weight
            # Calculate the squared sum of the remaining weights
            penalty = torch.sum(remaining_weights ** 2, dim=1)
            # Sum up the penalties for all neurons in the layer
            regularization_loss += torch.sum(penalty)

    total_penalty = lambda_sparsity * regularization_loss
    return total_penalty

def train_with_rectified_L2(model, loss_function, optimizer, x, y, 
                            no_of_epochs=90000, ALPHA=0.5,
                            LAMBDA_MAGNITUDE = 0.01,
                            LAMBDA_POLARITY = 0.01,
                            LAMBDA_INTEGERS = 0.01,
                            LAMBDA_SPARSITY = None,
                            initial_lr=0.01,
                            max_lr=1):
    # store loss and penalization for each epoch
    all_loss_without_reg = []
    all_loss_with_reg = []

    for epoch in range(no_of_epochs):
        # forward pass
        y_hat = model(x)

        # compute gradient G1 - loss
        loss = loss_function(y_hat, y)
        
        all_loss_without_reg.append(loss.item())

        # compute opposite sign penalty
        polarity_penalty = calculate_total_opposite_sign_penalty(model, LAMBDA_POLARITY)

        # compute same weight magnitude penalty
        magnitude_penalty = calculate_weight_magnitude_penalty(model, LAMBDA_MAGNITUDE)

        # compute distance from integer numbers penalty
        integer_penalty = calculate_integer_penalty2(model, LAMBDA_INTEGERS=LAMBDA_INTEGERS)

        if LAMBDA_SPARSITY is not None:
            sparsity_penalty = calculate_sparse_penalty(model, lambda_sparsity=LAMBDA_SPARSITY)
            # total loss with regularization
            if epoch > 1000:
                total_loss = loss + polarity_penalty + magnitude_penalty + integer_penalty + sparsity_penalty
            else:
                total_loss = loss + polarity_penalty + magnitude_penalty + sparsity_penalty
        else:
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
    LAMBDA_INTEGERS = 0.001
    LAMBDA_SPARSITY = 1

    # # dataset
    # X_train, y_train, X_test, y_test = create_torch_XOR_dataset()

    # model_XOR = TwoLayerMLP().double()
    # # define the loss
    # loss_function = torch.nn.BCELoss()

    # #optimizer = torch.optim.SGD(model_XOR.parameters(), lr=INIITIAL_LEARNING_RATE)
    # optimizer = torch.optim.Adam(model_XOR.parameters(), lr=INIITIAL_LEARNING_RATE, eps=1e-7)

    # all_loss = train_with_rectified_L2(model_XOR, 
    #                                 loss_function, 
    #                                 optimizer, 
    #                                 X_train, 
    #                                 y_train,
    #                                 no_of_epochs=EPOCHS,
    #                                 ALPHA=ALPHA,
    #                                 LAMBDA_MAGNITUDE=LAMBDA_MAGNITUDE,
    #                                 LAMBDA_POLARITY=LAMBDA_POLARITY,
    #                                 LAMBDA_INTEGERS=LAMBDA_INTEGERS,
    #                                 initial_lr=INIITIAL_LEARNING_RATE,
    #                                 max_lr=MAXIMUM_LEARNING_RATE)
    
    X_train, y_train, X_test, y_test = create_torch_XOR_XNOR_dataset()

    model_XOR_AND_XNOR = ThreeLayerMLP().double()
    # define the loss
    loss_function = torch.nn.BCELoss()
    #optimizer = torch.optim.SGD(model_XOR.parameters(), lr=INIITIAL_LEARNING_RATE)
    optimizer = torch.optim.Adam(model_XOR_AND_XNOR.parameters(), lr=INIITIAL_LEARNING_RATE, eps=1e-7)

    all_loss = train_with_rectified_L2(model_XOR_AND_XNOR, 
                                    loss_function, 
                                    optimizer, 
                                    X_train, 
                                    y_train,
                                    no_of_epochs=EPOCHS,
                                    ALPHA=ALPHA,
                                    LAMBDA_MAGNITUDE=LAMBDA_MAGNITUDE,
                                    LAMBDA_POLARITY=LAMBDA_POLARITY,
                                    LAMBDA_INTEGERS=LAMBDA_INTEGERS,
                                    LAMBDA_SPARSITY=LAMBDA_SPARSITY,
                                    initial_lr=INIITIAL_LEARNING_RATE,
                                    max_lr=MAXIMUM_LEARNING_RATE)


