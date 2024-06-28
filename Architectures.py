import torch
import torch.nn as nn

def make_mlp(input_dim, output_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='cpu'):
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers).to(device)


def make_cnn_sequential(input_dim, output_dim, hidden_dims=(32, 64), activation=nn.ReLU, output_activation=None):
    layers = []
    in_channels = input_dim[0]
    for h_dim in hidden_dims:
        layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=1, padding=1))
        layers.append(activation())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = h_dim
    layers.append(nn.Flatten())
    layers.append(nn.Linear(hidden_dims[-1] * (input_dim[1] // 2 ** len(hidden_dims)) ** 2, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def make_continuous_action_mlp(input_dim, output_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None):
    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation())
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)