from typing import Optional
import gymnasium
import numpy as np
import torch

from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params
from callbacks import AUCCallback
from utils import polyak
from Logger import WandBLogger, TensorboardLogger

import schedulefree as sf

class LogUVAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 beta: float = 1.0,
                 use_target_network: bool = False,
                 target_update_interval: Optional[int] = None,
                 polyak_tau: Optional[float] = None,
                 **kwargs,
                 ):
        
        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())
        
        self.algo_name = 'LogUV'
        self.gamma = gamma
        self.beta = beta
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval
        self.polyak_tau = polyak_tau

        self.nA = self.env.action_space.n
        self.log_hparams(self.kwargs)
        
        self.online_logus = self.architecture
        self.online_logvs = self.architecture

        if self.use_target_network:
            self.target_logus = self.architecture
            self.target_logus.load_state_dict(self.online_logus.state_dict())
            self.target_logvs = self.architecture
            self.target_logvs.load_state_dict(self.online_logvs.state_dict())
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
            self.target_logus = self.online_logus
            self.target_logvs = self.online_logvs
            # Raise a warning if update interval is specified:
            if target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")

        #alias the self.model as both online logus and logvs:
        # later experiment with using same backbone
        self.model = (self.online_logus, self.online_logvs)

        # Make (all) softqs learnable:
        # self.optimizer = sf.AdamWScheduleFree(self.model.parameters(), lr=self.learning_rate)
        self.u_opt = torch.optim.Adam(self.online_logus.parameters(), lr=self.learning_rate)
        self.v_opt = torch.optim.Adam(self.online_logvs.parameters(), lr=self.learning_rate)

        # TODO: allow for non uniform priors
        self.log_pi0 = -torch.log(torch.tensor(self.nA, device=self.device))

    def exploration_policy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            # return self.env.action_space.sample()
            qvals = self.online_logus(state)
            # calculate boltzmann policy:
            qvals = qvals.squeeze()
            # sample from logits:
            pi = torch.distributions.Categorical(logits = self.beta * qvals + self.log_pi0)
            action = pi.sample()
            return action.item()
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # self.optimizer.eval()
        # Get the greedy action from the q values:
        with torch.no_grad():
            qvals = self.online_logus(state) + 1 / self.beta * self.log_pi0
            qvals = qvals.squeeze()
            return torch.argmax(qvals).item()
        

    def calculate_u_loss(self, batch):
        states, actions, next_states, dones, env_rewards = batch
        # TODO: integrate env_rewards as extrinsic, with intrinsic rewards
        rewards = self.reward_model(states, actions)

        actions = actions.long()
        dones = dones.float()
        curr_softq = self.online_logus(states).squeeze().gather(1, actions)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_softqs = self.target_logus(next_states)
            
            next_v = 1/self.beta * (torch.logsumexp(self.beta * next_softqs + self.log_pi0, dim=-1) )
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)

        # Calculate the softq ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_softq, expected_curr_softq)
        
        self.log_history("train/online_q_mean", curr_softq.mean().item(), self.learn_env_steps)
        # log the loss:
        self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss
    
    def calculate_v_loss(self, batch):
        states, actions, next_states, dones, rewards = batch

        rewards = self.reward_model(states, actions)

        actions = actions.long()
        dones = dones.float()
        curr_softq = self.online_logvs(states).squeeze().gather(1, actions)
        with torch.no_grad():
            if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_softqs = self.target_logus(next_states)
            
            next_v = 1/self.beta * (torch.logsumexp(self.beta * next_softqs + self.log_pi0, dim=-1) )
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_softq = rewards + self.gamma * next_v * (1-dones)

        # Calculate the softq ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_softq, expected_curr_softq)
        
        self.log_history("train/online_q_mean", curr_softq.mean().item(), self.learn_env_steps)
        # log the loss:
        self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss
    

    def calculate_loss(self, batch):
        return self.calculate_u_loss(batch) + self.calculate_v_loss(batch)

    def reward_model(self, state: np.ndarray, action: int) -> float:
        # Returns log(uv) for max exploration:
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor([action], device=self.device, dtype=torch.int64)
            logu = self.online_logus(state).squeeze().gather(0, action)
            logv = self.online_logvs(state).squeeze().gather(0, action)
            return (logu + logv).item()

    def _on_step(self) -> None:
        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.online_softqs, self.target_softqs, self.polyak_tau, self.device)

        super()._on_step()


if __name__ == '__main__':
    # set the seed:
    torch.manual_seed(0)
    np.random.seed(0)

    import gymnasium as gym
    env = gym.make('CartPole-v1')
    logger = TensorboardLogger('logs/cp-sf')
    #logger = WandBLogger(entity='jacobhadamczyk', project='test')
    mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[64, 64],
                   activation=torch.nn.ReLU)
    agent = SoftQAgent(env,
                       architecture=mlp, 
                       loggers=(logger,),
                       learning_rate=0.016,
                       learning_starts=0,
                       gamma=0.99,
                       beta=0.02,
                       train_interval=2,
                       gradient_steps=1,
                       batch_size=512,
                       use_target_network=False,
                       target_update_interval=1,
                       polyak_tau=1.0,
                       eval_callbacks=[AUCCallback],
                       use_threaded_eval=True,
                       seed=0
                       )
    agent.learn(total_timesteps=50000)
