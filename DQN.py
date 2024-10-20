from typing import Optional
import gymnasium
import numpy as np
import torch
from Architectures import make_atari_nature_cnn, make_mlp
from BaseAgent import BaseAgent, get_new_params
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
        super()._on_step()

        # Update epsilon:
        self.epsilon = max(self.minimum_epsilon, (self.initial_epsilon - self.learn_env_steps / self.total_timesteps / self.exploration_fraction))

        if self.learn_env_steps % self.log_interval == 0:
            self.log_history("train/epsilon", self.epsilon, self.learn_env_steps)

        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.online_qs, self.target_qs, self.polyak_tau, self.device)


    def exploration_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.evaluation_policy(state)
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_qs(state)
        qvals = qvals.squeeze()
        return torch.argmax(qvals).cpu().item()
    

    def calculate_loss(self, batch):
        # states, actions, rewards, next_states, dones = batch
        states = batch.observations
        actions = batch.actions
        next_states = batch.next_observations
        dones = batch.dones
        rewards = batch.rewards
        actions = actions.long()
        dones = dones.float()
        curr_q = self.online_qs(states).squeeze().gather(1, actions.long())
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_qs = self.target_qs(next_states)
            
            next_v = torch.max(next_qs, dim=-1).values
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_q = rewards + self.gamma * next_v * (1-dones)

        # Calculate the q ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_q, expected_curr_q)
        
        self.log_history("train/online_q_mean", curr_q.mean().item(), self.learn_env_steps)
        # log the loss:
        self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss


if __name__ == '__main__':
    import gymnasium as gym
    env = 'ALE/Pong-v5'

    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/atari')

    # env = 'CartPole-v1'
    agent = DQN(env, 
                # architecture=make_mlp,
                # architecture_kwargs={'input_dim': gym.make(env).observation_space.shape[0],
                #                      'output_dim': gym.make(env).action_space.n,
                #                      'hidden_dims': [64, 64]},
                architecture=make_atari_nature_cnn,
                architecture_kwargs={'output_dim': gym.make(env).action_space.n},
                loggers=(logger,),
                learning_rate=0.0001,
                train_interval=4,
                gradient_steps=4,
                batch_size=64,
                use_target_network=True,
                target_update_interval=1_000,
                polyak_tau=1.0,
                learning_starts=50_000,
                log_interval=5000,
                use_threaded_eval=True,


                )
    agent.learn(total_timesteps=10_000_000)
