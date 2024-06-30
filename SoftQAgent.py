from typing import Optional
import gymnasium
import numpy as np
import torch

from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params, AUCCallback
from utils import polyak
from Logger import WandBLogger, TensorboardLogger


class SoftQAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 beta: float = 5.0,
                 use_target_network: bool = False,
                 target_update_interval: Optional[int] = None,
                 polyak_tau: Optional[float] = None,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'SQL'
        self.gamma = gamma
        self.beta = beta
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval
        self.polyak_tau = polyak_tau

        self.nA = self.env.action_space.n
        self.log_hparams(self.kwargs)
        
        self.online_softqs = self.architecture
        if self.use_target_network:
            self.target_softqs = self.architecture
            self.target_softqs.load_state_dict(self.online_softqs.state_dict())
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
            self.target_softqs = self.online_softqs
            # Raise a warning if update interval is specified:
            if target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")


        self.model = self.online_softqs

        # Make (all) softqs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def exploration_policy(self, state: np.ndarray) -> int:
        # return self.env.action_space.sample()
        qvals = self.online_softqs(torch.from_numpy(state).to(device=self.device))
        # calculate boltzmann policy:
        qvals = qvals.squeeze()
        # sample from logits:
        pi = torch.distributions.Categorical(logits = self.beta * qvals)
        action = pi.sample()
        return action.item()
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_softqs(torch.from_numpy(state)).to(device=self.device)
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = actions.unsqueeze(0).long()
        dones = dones.float()
        curr_softq = self.online_softqs(states).squeeze().gather(1, actions)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_softqs = self.target_softqs(next_states)
            
            next_v = 1/self.beta * (torch.logsumexp(
                self.beta * next_softqs, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v = next_v.unsqueeze(0)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)

        # Calculate the softq ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_softq, expected_curr_softq)
        
        self.log_history("train/online_q_mean", curr_softq.mean().item(), self.learn_env_steps)
        # log the loss:
        self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss

    def _on_step(self) -> None:
        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.online_softqs, self.target_softqs, self.polyak_tau)

        super()._on_step()


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    logger = TensorboardLogger('logs/cartpole')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32])
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
                       polyak_tau=1.0,
                       eval_callbacks=[AUCCallback],
                       )
    agent.learn(total_timesteps=50000)
