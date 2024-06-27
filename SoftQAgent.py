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
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'SQL'
        self.gamma = gamma
       
        self.log_hparams(self.kwargs)
        

        self.online_softqs = self.architecture
            
        self.model = self.online_softqs

        # Make (all) softqs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        qvals = self.online_softqs(state)
        # calculate boltzmann policy:
        qvals = qvals.squeeze()
        qvals = qvals - torch.max(qvals)
        qvals = qvals/self.beta
        qvals = torch.exp(qvals)
        qvals = qvals/torch.sum(qvals)
        return torch.multinomial(qvals, 1).item()
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_softqs(state)
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def gradient_descent(self, batch):
        states, actions, next_states, dones, rewards = batch
        curr_softq = torch.stack([softq(states).squeeze().gather(1, actions.long())
                                for softq in self.online_softqs], dim=0)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()
            
            online_curr_softq = torch.stack([softq(states).gather(1, actions)
                                            for softq in self.online_softqs], dim=0)

            online_curr_softq = online_curr_softq.squeeze(-1)

            next_softqs = [target_softq(next_states)
                                 for target_softq in self.target_softqs]
            next_softqs = torch.stack(next_softqs, dim=0)


            # aggregate the target next softqs:
            next_softq = self.aggregator_fn(next_softqs, dim=0)
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * next_softq, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)
            expected_curr_softq = expected_curr_softq.squeeze(1)

        # num_nets, batch_size, 1 (leftover from actions)
        curr_softq = curr_softq.squeeze(2)

        self.logger.record("train/online_q_mean", curr_softq.mean().item())

        # Calculate the softq ("critic") loss:
        loss = 0.5*sum(self.loss_fn(softq, expected_curr_softq)
                       for softq in curr_softq)
        
        # log the loss:
        self.logger.log("train/loss", loss.item())
        return loss


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[128, 128])
    agent = SoftQAgent(env, architecture=mlp)
    agent.learn(total_timesteps=100)
