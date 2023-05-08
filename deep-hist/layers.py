import torch
from torch import nn, sigmoid

def activation(x, L, W):
    x = x + (L / 2)
    sig_shift_1 = sigmoid(x / W)
    sig_shift_2 = sigmoid((x - L) / W)
    return sig_shift_1 - sig_shift_2

def compute_activation(x, mu, K, L, W):
    # flatten x
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return activation(x - mu, L, W)

class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()

        # initialize parameters
        self.k = 256
        self.L, self.W, self.mu = self.initialize_parameters(self.k)

    def initialize_parameters(self, k):
        L = 1 / k
        W = L / 2.5
        mu = (L * (torch.arange(k) + 0.5)).view(-1, 1)
        return L, W, mu

class SingleDimensionalHistLayer(BaseLayer):
    def __init__(self):
        super(SingleDimensionalHistLayer, self).__init__()

    def forward(self, x):
        num_elems = x.size(1) * x.size(2)
        activation_bins = compute_activation(x, self.mu, self.k, self.L, self.W)
        return activation_bins.sum(dim=2) / num_elems

class TwoDimensionalHistLayer(BaseLayer):
    def __init__(self):
        super(TwoDimensionalHistLayer, self).__init__()

    def forward(self, x, y):
        num_elems = x.size(1) * x.size(2)
        activation_bins_x = compute_activation(x, self.mu, self.k, self.L, self.W)
        activation_bins_y = compute_activation(y, self.mu, self.k, self.L, self.W)
        act = torch.matmul(activation_bins_x, activation_bins_y.transpose(1, 2)) / num_elems
        return act