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
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim).to(self.device))
            self.fc_layers.append(activation())
            in_dim = hidden_dim
        self.fc_layers.append(nn.Linear(in_dim, output_dim).to(self.device))
        if output_activation is not None:
            self.fc_layers.append(output_activation())

        self.fc_layers = nn.Sequential(*self.fc_layers).to(self.device)
        

    def forward(self, x):
        x = preprocess_obs(x, device=self.device)  # Apply preprocessing
        # Add batch dimension if needed (handle unbatched inputs like shape (4,))
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.fc_layers(x)
        return x

class EnsembleMLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 n_networks: int = 2,
                 *args, 
                 activation=nn.ReLU,
                 hidden_dims=(64, 64), 
                 output_activation=None,
                 ensemble_aggregation=lambda x: torch.min(x, dim=0)[0],
                 device='auto',
                 **kwargs) -> None:
        super(EnsembleMLP, self).__init__()
        self.device = auto_device(device)
        self.n_networks = n_networks
        self.ensemble_aggregation = ensemble_aggregation
        self.models = nn.ModuleList([
            MLP(input_dim, output_dim, 
                activation=activation, 
                hidden_dims=hidden_dims, 
                output_activation=output_activation,
                device=device)
            for _ in range(n_networks)
        ]).to(self.device)

    def forward(self, x):
        # Don't preprocess here - let each MLP handle it
        outputs = [model(x) for model in self.models]
        aggregated = self.ensemble_aggregation(torch.stack(outputs, dim=0))
        return aggregated

class ConcatInputMLP(MLP):
    def __init__(self, obs_dim, action_dim, output_dim, *args, **kwargs):
        super().__init__(input_dim=obs_dim + action_dim, output_dim=output_dim, *args, **kwargs)

    def forward(self, obs, action): # TODO: extend to arbitrary number of inputs
        x = torch.cat([obs, action], dim=-1)
        return super().forward(x)

def make_mlp(input_dim=None, output_dim=None, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='auto'):
    return MLP(input_dim, 
               output_dim, 
               hidden_dims=hidden_dims, 
               activation=activation, 
               output_activation=output_activation, 
               device=device)

def make_min_continuous_action_critic(obs_dim, action_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='auto'):
    # creates an ensemble of 2 critics and takes the min Q value
    return EnsembleMLP(input_dim=obs_dim + action_dim, 
                       output_dim=1, 
                       n_networks=2,
                       hidden_dims=hidden_dims, 
                       activation=activation, 
                       output_activation=output_activation,
                       ensemble_aggregation=lambda x: torch.min(x, dim=0)[0],
                       device=device)

def make_min_discrete_action_critic(obs_dim, n_actions, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None, device='auto'):
    # creates an ensemble of 2 critics and takes the min Q value
    return EnsembleMLP(input_dim=obs_dim, 
                       output_dim=n_actions, 
                       n_networks=2,
                       hidden_dims=hidden_dims, 
                       activation=activation, 
                       output_activation=output_activation,
                       ensemble_aggregation=lambda x: torch.min(x, dim=0)[0],
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
        ).to(self.device)

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
        x = x.to(self.device)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def make_atari_nature_cnn(output_dim, input_dim=(84, 84, 4), device='auto', activation=nn.ReLU, hidden_dim=512):
    model = AtariNatureCNN(output_dim, input_dim, device, activation, hidden_dim)
    return model


def make_sac_critic_mlp(obs_dim, action_dim, hidden_dims=(128, 128), activation=nn.ReLU, output_activation=None):
    # cat's the state and action inputs together then passes into an MLP
    return ConcatInputMLP(obs_dim, 
                          action_dim, 
                          output_dim=1,
                          hidden_dims=hidden_dims, 
                          activation=activation, 
                          output_activation=output_activation)

class DummyActor(nn.Module):
    def __init__(self, obs_dim, action_dim, device='auto'):
        super(DummyActor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = auto_device(device)

    def forward(self, state, deterministic=True):
        raise NotImplementedError("DummyActor does not implement forward().")
    
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(128, 128), activation=nn.ReLU, log_std_min=-20, log_std_max=2, device='auto'):
        super(GaussianActor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = auto_device(device)

        # Need to output mean and log_std for each action_dim
        self.net = MLP(obs_dim, action_dim * 2, hidden_dims, activation).to(self.device)

    def forward(self, state, deterministic=False):
        state = preprocess_obs(state, device=self.device)
        mean_logstd = self.net(state)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        log_prob = 0.0 # deterministic samples have prob=1
        if deterministic:
            action = mean
        else:
            # Compute log probability
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # Reparameterization trick
            # TODO: should be equivalent to mean + std * N(0,1)
            log_prob = normal.log_prob(action).sum(axis=-1, keepdim=True)
            # TODO: implement squashing correction and flag
            # Apply squashing function (tanh) and adjust log prob
            action = torch.tanh(action)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1, keepdim=True)

        return action, log_prob
    
def make_gaussian_actor(obs_dim, action_dim, hidden_dims=(128, 128), activation=nn.ReLU, log_std_min=-20, log_std_max=2, device='auto'):
    return GaussianActor(obs_dim, 
                         action_dim, 
                         hidden_dims, 
                         activation, 
                         log_std_min, 
                         log_std_max, 
                         device)