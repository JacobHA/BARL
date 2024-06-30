import torch
import torch.nn as nn
from utils import auto_device

def make_mlp(input_dim, output_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='auto'):
    device = auto_device(device)
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

def make_atari_nature_cnn(output_dim, input_dim=(84,84,4), device='auto', activation=nn.ReLU, hidden_dim=512):
    device = auto_device(device)
    layers = []
    # Use a CNN:
    n_channels = input_dim[2]
    layers += [
        nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, device=device),
        activation(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, device=device),
        activation(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, device=device),
        nn.Flatten(start_dim=1),
        ]
        # calculate resulting shape for FC layers:
    with torch.no_grad():
        rand_inp = torch.rand(1, *input_dim)
        x = rand_inp.to(device)  # Convert to PyTorch tensor
        x = x.detach()
        # x = preprocess_obs(x, observation_space, normalize_images=NORMALIZE_IMG)
        x = x.permute(0,3,1,2)
        flat_size = nn.Sequential(*layers)(x).shape[1]

    print(f"Using a CNN with {flat_size}-dim. flattened output.")

    layers += [
        nn.Linear(flat_size, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    ]
    print(f"Parameter count: {sum(p.numel() for p in nn.Sequential(*layers).parameters())}")

    return nn.Sequential(*layers).to(device)


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