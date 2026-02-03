from typing import Optional
import gymnasium
import numpy as np
import torch
from Architectures import make_atari_nature_cnn, make_mlp
from BaseAgent import BaseAgent, get_new_params
from network_monitor import NetworkMonitorCallback, create_monitor_for_agent
from utils import polyak


class DQN(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 minimum_epsilon: float = 0.05,
                 exploration_fraction: float = 0.5,
                 initial_epsilon: float = 1.0,
                 use_target_network: bool = False,
                 target_update_interval: Optional[int] = None,
                 polyak_tau: Optional[float] = None,
                 architecture_kwargs: dict = {},
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'DQN'
        self.gamma = gamma
        self.minimum_epsilon = minimum_epsilon
        self.exploration_fraction = exploration_fraction
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval
        self.polyak_tau = polyak_tau
       
        self.nA = self.env.action_space.n
        # Add algo_name and env_str to kwargs for logging
        self.kwargs['algo_name'] = self.algo_name
        self.kwargs['env_str'] = self.env_str
        self.log_hparams(self.kwargs)
        self.online_qs = self.architecture(**architecture_kwargs)
        self.model = self.online_qs

        if self.use_target_network:
            # Make another instance of the architecture for the target network:
            self.target_qs = self.architecture(**architecture_kwargs)
            self.target_qs.load_state_dict(self.online_qs.state_dict())
            if polyak_tau is not None:
                assert 0 <= polyak_tau <= 1, "Polyak tau must be in the range [0, 1]."
                self.polyak_tau = polyak_tau
            else:
                print("WARNING: No polyak tau specified for soft target updates. Using default tau=1 for hard updates.")
                self.polyak_tau = 1.0

            if target_update_interval is None:
                print("WARNING: Target network update interval not specified. Using default interval of 1 step.")
                self.target_update_interval = 1
        # Alias the "target" with online net if target is not used:
        else:
            self.target_qs = self.online_qs
            # Raise a warning if update interval is specified:
            if target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")

        # Make (all) qs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _on_step(self) -> None:

        # Update epsilon:
        progress = self.learn_env_steps / max(1, self.total_learn_env_steps)
        self.epsilon = max(self.minimum_epsilon, self.initial_epsilon - progress / self.exploration_fraction)

        if self.learn_env_steps % self.log_interval == 0:
            self.log_history("train/epsilon", self.epsilon, self.learn_env_steps)

        # Periodically update the target net
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.online_qs, self.target_qs, self.polyak_tau)

        super()._on_step()


    def exploration_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.evaluation_policy(state)
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        # Reshape to batch format: (4,) -> (1, 4)
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            qvals = self.online_qs(state)
        return torch.argmax(qvals[0]).item()
    
    def gradient_step(self, grad_step):
        # Sample a batch from the replay buffer:
        batch = self.buffer.sample(self.batch_size)

        loss = self.calculate_loss(batch)
        self.optimizer.zero_grad()

        # Clip gradient norm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = actions.long()
        dones = dones.float()
        
        # Forward pass through online network
        q_values = self.online_qs(states)  # (batch_size, action_size)
        curr_q = q_values.gather(1, actions)  # (batch_size, 1)
        
        with torch.no_grad():
            # Forward pass through target network for next states
            next_q_values = self.target_qs(next_states)  # (batch_size, action_size)
            next_v = torch.max(next_q_values, dim=-1).values  # (batch_size,)
            next_v = next_v.reshape(-1, 1)  # (batch_size, 1)

            # Backup equation:
            expected_curr_q = rewards + self.gamma * next_v * (1 - dones)

        # Calculate the q ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_q, expected_curr_q)
        
        self.log_history("train/online_q_mean", curr_q.mean().item(), self.learn_env_steps)
        # log the loss:
        self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss
    
    def save_model(self, filepath: str):
        """Save the model to the specified filepath."""
        torch.save({
            'online_qs_state_dict': self.online_qs.state_dict(),
            'target_qs_state_dict': self.target_qs.state_dict() if self.use_target_network else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)


if __name__ == '__main__':
    import gymnasium as gym
    env = 'ALE/Pong-v5'

    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/atari')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    # mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32])#, activation=torch.nn.Mish)
    # cnn = make_atari_nature_cnn(gym.make(env).action_space.n)
    # Create monitor configured for your agent type
    monitor, networks = create_monitor_for_agent(
        named_networks=["online_qs"],
        log_frequency=100,  # Log every 100 gradient steps
        track_eigenvalues=True,  # Expensive, keep disabled
    )

    callback = NetworkMonitorCallback(monitor, networks)
    env = 'Acrobot-v1'
    agent = DQN(env, 
                architecture=make_mlp,
                architecture_kwargs={'input_dim': gym.make(env).observation_space.shape[0],
                                     'output_dim': gym.make(env).action_space.n,
                                     'hidden_dims': [128, 128]},
                loggers=(logger,),
                learning_rate=0.003,
                exploration_fraction=0.05,
                initial_epsilon=1.0,
                minimum_epsilon=0.08,
                train_interval=10,
                gradient_steps=4,
                batch_size=64,
                use_target_network=True,
                target_update_interval=10,
                polyak_tau=1.0,
                learning_starts=5000,
                log_interval=500,
                record_eval_video=True,
                eval_video_every=5,
                eval_video_async=True,
                network_monitor=callback,  # <-- Add monitoring
                )

    agent.learn(total_timesteps=160_000)
