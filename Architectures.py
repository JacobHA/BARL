import numpy as np
import torch
import torch.nn as nn
from utils import auto_device

class MLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 *args, 
                 activation=nn.ReLU,
                 hidden_dims=(64, 64), 
                 output_activation=None,
                 device='auto', 
                 **kwargs) -> None:
        super(MLP, self).__init__()
        self.device = auto_device(device)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.output_activation = output_activation

        self.fc_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim, device=self.device))
            self.fc_layers.append(activation())
            in_dim = hidden_dim
        self.fc_layers.append(nn.Linear(in_dim, output_dim, device=self.device))
        if output_activation is not None:
            self.fc_layers.append(output_activation())

        self.fc_layers = nn.Sequential(*self.fc_layers, device=self.device)
        

    def forward(self, x):
        x = preprocess_obs(x, device=self.device)  # Apply preprocessing
        x = self.fc_layers(x)
        return x

    

def make_mlp(input_dim=None, output_dim=None, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='auto'):
    return MLP(input_dim, 
               output_dim, 
               hidden_dims=hidden_dims, 
               activation=activation, 
               output_activation=output_activation, 
               device=device)


def make_cnn_sequential(input_dim, output_dim, hidden_dims=(32, 64), activation=nn.ReLU, output_activation=None):
    layers = []
    in_channels = input_dim[0]
    for hidden_dim in hidden_dims:
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1))
        layers.append(activation())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        in_channels = hidden_dim
    layers.append(nn.Flatten())
    layers.append(nn.Linear(hidden_dims[-1] * (input_dim[1] // 2 ** len(hidden_dims)) ** 2, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def preprocess_obs(obs, device):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    # Pixel observations
    if obs.dtype == torch.uint8:
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0).to(device=device)
            # Normalize pixel values to the range [0, 1]
        return obs.float() / 255.0
    if len(obs.shape) == 4:
        # Change to (N, C, H, W) format
        obs = obs.permute(0, 3, 1, 2).to(device)  

        # Normalize pixel values to the range [0, 1]
        return obs.float() / 255.0
    return obs.to(device=device)

class AtariNatureCNN(nn.Module):
    def __init__(self, output_dim, input_dim=(84, 84, 4), device='auto', activation=nn.ReLU, hidden_dim=512):
        super(AtariNatureCNN, self).__init__()
        self.device = auto_device(device)
        
        n_channels = input_dim[2]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, device=self.device),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, device=self.device),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, device=self.device),
            activation(),
            nn.Flatten(start_dim=1)
        )

        # Calculate resulting shape for FC layers:
        with torch.no_grad():
            rand_inp = torch.rand(1, *input_dim)
            rand_inp = preprocess_obs(rand_inp, device=self.device)  # Preprocess the random input
            flat_size = self.conv_layers(rand_inp).shape[1]

        print(f"Using a CNN with {flat_size}-dim. flattened output.")

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, hidden_dim, device=self.device),
            activation(),
            nn.Linear(hidden_dim, output_dim, device=self.device)
        )
        print(f"Parameter count: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x):
        x = preprocess_obs(x, device=self.device)  # Apply preprocessing
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def make_atari_nature_cnn(output_dim, input_dim=(84, 84, 4), device='auto', activation=nn.ReLU, hidden_dim=512):
    model = AtariNatureCNN(output_dim, input_dim, device, activation, hidden_dim)
    return model


def make_continuous_action_mlp(input_dim, output_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None):
    layers = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)