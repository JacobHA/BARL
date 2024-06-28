import gymnasium
import numpy as np
import torch
from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params
from utils import logger_at_folder

class SoftQAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 beta: float = 5.0,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.beta = beta
       
        self.nA = self.env.action_space.n
        self.log_hparams(self.kwargs)
        
        self.online_softqs = self.architecture
            
        self.model = self.online_softqs

        # Make (all) softqs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        qvals = self.online_softqs(torch.tensor(state))
        # calculate boltzmann policy:
        qvals = qvals.squeeze()
        qvals = qvals - torch.max(qvals)
        qvals = qvals/self.beta
        qvals = torch.exp(qvals)
        qvals = qvals/torch.sum(qvals)
        return torch.multinomial(qvals, 1).item()
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_softqs(torch.tensor(state))
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def calculate_loss(self, batch):
        states, actions, next_states, dones, rewards = batch
        curr_softq = self.online_softqs(states).squeeze().gather(1, actions.long())
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()
            
            online_curr_softq = self.online_softqs(states).gather(1, actions)

            online_curr_softq = online_curr_softq.squeeze(-1)

            next_softqs = self.online_softqs(next_states)
            
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * next_softqs, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)
            # expected_curr_softq = expected_curr_softq.squeeze(1)


        # Calculate the softq ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_softq, expected_curr_softq)
        
        for logger in self.loggers:
            logger.log_history("train/online_q_mean", curr_softq.mean().item(), self.env_steps)
            # log the loss:
            logger.log_history("train/loss", loss.item(), self.env_steps)

        return loss


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/cartpole')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32])
    agent = SoftQAgent(env, architecture=mlp, loggers=(logger,), max_grad_norm=10)
    agent.learn(total_timesteps=50000)
