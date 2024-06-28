import gymnasium
import numpy as np
import torch
from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params
from utils import logger_at_folder

class DQN(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 minimum_epsilon: float = 0.05,
                 exploration_fraction: float = 0.5,
                 initial_epsilon: float = 1.0,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.minimum_epsilon = minimum_epsilon
        self.exploration_fraction = exploration_fraction
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
       
        self.nA = self.env.action_space.n
        self.log_hparams(self.kwargs)
        
        self.online_qs = self.architecture
            
        self.model = self.online_qs

        # Make (all) qs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _on_step(self) -> None:
        super()._on_step()
        # Update epsilon:
        self.epsilon = max(self.minimum_epsilon, (self.initial_epsilon - self.env_steps / self.total_timesteps / self.exploration_fraction))

        if self.env_steps % self.log_interval == 0:
            for logger in self.loggers:
                logger.log_history("train/epsilon", self.epsilon, self.env_steps)

    def exploration_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.evaluation_policy(state)
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_qs(torch.tensor(state))
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def calculate_loss(self, batch):
        states, actions, next_states, dones, rewards = batch
        curr_q = self.online_qs(states).squeeze().gather(1, actions.long())
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()
            
            online_curr_q = self.online_qs(states).gather(1, actions)

            online_curr_q = online_curr_q.squeeze(-1)

            next_qs = self.online_qs(next_states)
            
            next_v = torch.max(next_qs, dim=-1).values
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_q = rewards + self.gamma * next_v * (1-dones)

        # Calculate the q ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_q, expected_curr_q)
        
        for logger in self.loggers:
            logger.log_history("train/online_q_mean", curr_q.mean().item(), self.env_steps)
            # log the loss:
            logger.log_history("train/loss", loss.item(), self.env_steps)

        return loss


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/cartpole')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32])#, activation=torch.nn.Mish)
    agent = DQN(env, 
                       architecture=mlp, 
                       loggers=(logger,),
                       learning_rate=0.001,
                       train_interval=1,
                       gradient_steps=1,
                       batch_size=256,
                       )
    agent.learn(total_timesteps=50000)
