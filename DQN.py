from typing import Optional
import gymnasium
import numpy as np
import torch
from Architectures import make_atari_nature_cnn, make_min_discrete_action_critic, make_mlp
from BaseAgent import BaseAgent, get_new_params
from Buffer import RNDUniformBuffer
from network_monitor import NetworkMonitorCallback, create_monitor_for_agent
from utils import check_polyak_tau, polyak, prepare_online_and_target


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
        
        check_polyak_tau(self.use_target_network, self.polyak_tau, self.target_update_interval)
        self.online_qs, self.target_qs = prepare_online_and_target(
            use_target_network=self.use_target_network,
            architecture=self.architecture,
            architecture_kwargs=architecture_kwargs)
        
        # Make (all) qs learnable:
        self.optimizer = torch.optim.Adam(self.online_qs.parameters(), lr=self.learning_rate)

    def _on_step(self) -> None:
        # Update epsilon:
        progress = self.learn_env_steps / max(1, self.total_learn_env_steps)
        self.epsilon = max(self.minimum_epsilon, self.initial_epsilon - progress / self.exploration_fraction)

        if self.learn_env_steps % self.log_interval == 0:
            self.log_history("train/epsilon", self.epsilon, self.learn_env_steps)

        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.target_qs, self.online_qs, self.polyak_tau)

        super()._on_step()

    def exploration_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.evaluation_policy(state)

    def evaluation_policy(self, state: np.ndarray) -> int:       
        with torch.no_grad():
            qvals = self.online_qs(state)
            # If output has batch dimension (from MLP adding it), remove it
            if len(qvals.shape) == 2:
                qvals = qvals.squeeze(0)
        return torch.argmax(qvals).item()

    def gradient_step(self, grad_step):
        # Sample a batch from the replay buffer:
        batch = self.buffer.sample(self.batch_size)
        should_log = (grad_step == 0) and (self.learn_env_steps % self.log_interval == 0)
        loss = self.calculate_loss(batch, log_metrics=should_log)
        self.optimizer.zero_grad()
        # Clip gradient norm
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online_qs.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def calculate_loss(self, batch, log_metrics: bool = True):
        states, actions, rewards, next_states, dones = batch
        actions = actions.long()
        dones = dones.float()
        
        # Ensure actions have shape (batch_size, 1) for gather
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        
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
        loss = torch.nn.functional.mse_loss(curr_q, expected_curr_q)
        
        if log_metrics:
            self.log_history("train/online_q_mean", curr_q.mean().item(), self.learn_env_steps)
            self.log_history("train/online_q_std", curr_q.std().item(), self.learn_env_steps)
            self.log_history("train/online_q_min", curr_q.min().item(), self.learn_env_steps)
            self.log_history("train/online_q_max", curr_q.max().item(), self.learn_env_steps)
            self.log_history("train/target_q_mean", expected_curr_q.mean().item(), self.learn_env_steps)
            self.log_history("train/target_q_std", expected_curr_q.std().item(), self.learn_env_steps)
            self.log_history("train/reward_mean", rewards.mean().item(), self.learn_env_steps)
            self.log_history("train/next_v_mean", next_v.mean().item(), self.learn_env_steps)
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
    logger = TensorboardLogger('logs/barl')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    # mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32])#, activation=torch.nn.Mish)
    tmp_env = gym.make(env)
    n_actions = tmp_env.action_space.n
    tmp_env.close()
    # Create monitor configured for your agent type
    monitor, networks = create_monitor_for_agent(
        named_networks=["online_qs"],
        log_frequency=100,  # Log every 100 gradient steps
        track_eigenvalues=True,  # Expensive, keep disabled
    )

    callback = NetworkMonitorCallback(monitor, networks)
    agent = DQN(env, 
                architecture=make_atari_nature_cnn,
                architecture_kwargs={'output_dim': n_actions},
                # architecture_kwargs={'input_dim': gym.make(env).observation_space.shape[0],
                #                      'output_dim': gym.make(env).action_space.n,
                #                      'hidden_dims': [32, 32]},
                loggers=(logger,),
                learning_rate=0.0003,
                gamma=0.99,
                exploration_fraction=0.16,
                initial_epsilon=1.0,
                minimum_epsilon=0.04,
                train_interval=4,
                gradient_steps=4,
                batch_size=16,
                use_target_network=True,
                target_update_interval=10_000,
                polyak_tau=1.0,
                learning_starts=50_000,
                log_interval=5000,
                record_eval_video=True,
                eval_video_every=5,
                # network_monitor=callback,  # <-- Add monitoring
                )

    agent.learn(total_timesteps=1000_000)

