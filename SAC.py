from typing import Optional
import numpy as np
import torch
from BaseAgent import BaseAgent, get_new_params
from network_monitor import NetworkMonitorCallback, create_monitor_for_agent
from utils import polyak
from Architectures import DummyActor, make_gaussian_actor, make_mlp, make_sac_critic_mlp

# TODO There is a big question about how to correctly specify different architectures. how do we allow for shared backbones?

class SAC(BaseAgent):
    def __init__(self,
                 *args,
                 alpha: float = 1.0,
                 gamma: float = 0.99,
                 use_target_network: bool = False,
                 target_update_interval: int = 1,
                 polyak_tau: float = 1.0,
                 actor_learning_rate: Optional[float] = None,
                 architecture_kwargs: dict = {},
                 **kwargs,
                 ):

        super().__init__(*args, **kwargs)
        self.kwargs = get_new_params(self, locals())

        self.algo_name = 'SAC'
        self.alpha = alpha
        self.gamma = gamma
        self.use_target_network = use_target_network
        self.target_update_interval = target_update_interval
        self.polyak_tau = polyak_tau
        self.actor_learning_rate = actor_learning_rate if actor_learning_rate is not None else self.learning_rate

        if isinstance(self.architecture, list):
            print("A list of architectures is provided. Unpacking as [Actor, Critic].")
            self.actor, self.critic = [arch(**kwargs) for arch, kwargs in zip(self.architecture, architecture_kwargs)]
        else:
            raise NotImplementedError("SAC requires a list of architectures [Actor, Critic].")

        # Ensure the actor network forward method has a deterministic kwarg:
        # assert isinstance(self.actor, DummyActor), "Actor must inherit from DummyActor class"
        # assert 'deterministic' in self.actor.forward.__kwargs__, "Actor forward pass must have deterministic kwarg"

        self.nA = self.env.action_space.shape[0]
        # TODO: Later use state action spaces to check the architecture

        # Ensure algo/env are present in logged hparams for dashboard display
        self.kwargs["algo_name"] = self.algo_name
        self.kwargs["env_str"] = self.env_str
        self.log_hparams(self.kwargs)

        if self.use_target_network:
            # Use a target critic
            # TODO: Toggle use of target actor
            self.target_critic = self.architecture[1](**architecture_kwargs[1]) # [Actor, Critic]
            self.target_critic.load_state_dict(self.critic.state_dict())
            # self.target_actor.load_state_dict(self.actor.state_dict())
            self.polyak_tau = polyak_tau
            if target_update_interval is None:
                print("WARNING: Target network update interval not specified. Using default interval of 1 step.")
                self.target_update_interval = 1

        # Alias the "target" with online net if target is not used:
        else:
            self.target_critic = self.critic
            # Raise a warning if update interval is specified:
            if target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)


    def _on_step(self) -> None:
        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.target_critic, self.critic, self.polyak_tau)
        super()._on_step()



    def exploration_policy(self, state: np.ndarray) -> int:
        # Return a stochastic sample from the actor network:
        with torch.no_grad():
            action, log_prob = self.actor(state, deterministic=False)
            return action.cpu().numpy()


    def evaluation_policy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            # Get the greedy action from the actor network:
            action, log_prob = self.actor(state, deterministic=True)
            return action.cpu().numpy()


    def calculate_critic_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        dones = dones.float()
        curr_q = self.critic(states, actions)

        with torch.no_grad():
            # Use Bellman backup equation for expected q value:
            # In SAC, we need V(s') = logsumexp(Q(s',a))
            # The integral is intractable so it is replaced with a single action
            next_actions, next_log_prob = self.actor(next_states, deterministic=False)
            next_q = self.target_critic(next_states, next_actions)
            # add soft policy contrib:
            next_v = next_q - self.alpha * next_log_prob

            expected_curr_q = rewards + self.gamma * next_v * (1 - dones)

        # Calculate the q ("critic") loss:
        critic_loss = 0.5*torch.nn.functional.mse_loss(curr_q, expected_curr_q)
        

        self.log_history("train/online_q_mean", curr_q.mean().item(), self.learn_env_steps)
        self.log_history("train/critic_loss", critic_loss.item(), self.learn_env_steps)


        return critic_loss
    
    def calculate_actor_loss(self, batch):
        states, _, _, _, _ = batch

        actions, log_prob = self.actor(states, deterministic=False)
        q_values = self.critic(states, actions)

        # Actor loss is based on minimizing KL for pi \propto exp(Q(s,a)/alpha)
        actor_loss = (self.alpha * log_prob - q_values).mean()
        self.log_history("train/actor_loss", actor_loss.item(), self.learn_env_steps)

        return actor_loss

    def gradient_step(self, grad_step):
        batch = self.buffer.sample(self.batch_size)
        critic_loss = self.calculate_critic_loss(batch)
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        actor_loss = self.calculate_actor_loss(batch)
        # Update actor network, using the new critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

    def save_checkpoint(self, filepath: str):
        """Save the model to the specified filepath."""
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict() if self.use_target_network else None,
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, filepath)

if __name__ == "__main__":

    from Logger import WandBLogger, TensorboardLogger
    logger = TensorboardLogger('logs/baseline')
    import gymnasium as gym
    monitor, networks = create_monitor_for_agent(
        named_networks=["critic", "actor"],
        log_frequency=100,  # Log every 100 gradient steps
        track_eigenvalues=True,  # Expensive, keep disabled
    )

    callback = NetworkMonitorCallback(monitor, networks)
    env = "Pendulum-v1"
    agent = SAC(env, 
                architecture=[make_gaussian_actor, make_sac_critic_mlp],
                architecture_kwargs=[{'obs_dim': gym.make(env).observation_space.shape[0], # actor takes a state and outputs a mean action
                                     'action_dim': gym.make(env).action_space.shape[0],
                                     'hidden_dims': [256, 256]},
                                     {'obs_dim': gym.make(env).observation_space.shape[0],
                                     'action_dim': gym.make(env).action_space.shape[0],
                                     'hidden_dims': [256, 256]},
                ],
                loggers=(logger,),
                learning_rate=0.003,
                alpha=0.2,
                train_interval=1,
                gradient_steps=1,
                batch_size=256,
                use_target_network=True,
                target_update_interval=1,
                polyak_tau=0.005,
                learning_starts=500,
                log_interval=500,
                record_eval_video=True,
                eval_video_every=5,
                network_monitor=callback,  # <-- Add monitoring
                )

    agent.learn(total_timesteps=160_000)