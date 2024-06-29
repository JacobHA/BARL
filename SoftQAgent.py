from typing import Optional
import gymnasium
import numpy as np
import torch
from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params, AUCCallback
from utils import logger_at_folder


class SoftQAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 beta: float = 5.0,
                 use_target_network: bool = False,
                 target_update_interval: Optional[int] = None,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.beta = beta
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval

        self.nA = self.env.action_space.n
        self.log_hparams(self.kwargs)
        
        self.online_softqs = self.architecture
        if self.use_target_network:
            self.target_softqs = self.architecture
            self.target_softqs.load_state_dict(self.online_softqs.state_dict())

        # Alias the "target" with online net if target is not used:
        else:
            self.target_softqs = self.online_softqs
            # Raise a warning if update interval is specified:
            if target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")


        self.model = self.online_softqs

        # Make (all) softqs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        qvals = self.online_softqs(torch.tensor(state, device=self.device))
        # calculate boltzmann policy:
        qvals = qvals.squeeze()
        # sample from logits:
        pi = torch.distributions.Categorical(logits = self.beta * qvals)
        action = pi.sample()
        return action.item()
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_softqs(torch.tensor(state, device=self.device))
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = actions.unsqueeze(1).long()
        dones = dones.float()
        curr_softq = self.online_softqs(states).squeeze().gather(1, actions)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_softqs = self.target_softqs(next_states)
            
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * next_softqs, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)

        # Calculate the softq ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_softq, expected_curr_softq)
        
        for logger in self.loggers:
            logger.log_history("train/online_q_mean", curr_softq.mean().item(), self.env_steps)
            # log the loss:
            logger.log_history("train/loss", loss.item(), self.env_steps)

        return loss

    def _on_step(self) -> None:
        # Periodically update the target network:
        if self.use_target_network and self.env_steps % self.target_update_interval == 0:
            self.target_softqs.load_state_dict(self.online_softqs.state_dict())
        super()._on_step()


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/cartpole')
    device= 'c'
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32], device='cuda')
    agent = SoftQAgent(env, 
                       architecture=mlp, 
                       loggers=(logger,),
                       learning_rate=0.001,
                       beta=0.5,
                       train_interval=1,
                       gradient_steps=1,
                       batch_size=256,
                       use_target_network=True,
                       target_update_interval=10,
                       eval_callbacks=[AUCCallback],
                       device='cuda'
                       )
    agent.learn(total_timesteps=50000)
