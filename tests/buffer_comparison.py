import gymnasium as gym
import sys

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

sys.path.append('./')
from Architectures import make_mlp
from BaseAgent import BaseAgent, get_new_params, AUCCallback
from Logger import WandBLogger, TensorboardLogger
from SoftQAgent import SoftQAgent


env = gym.make('CartPole-v1')
logger = TensorboardLogger('logs/cartpole')

class SB3_Buff_Adapter:
    def __init__(self, buffer):
        self.buffer = buffer
    def sample(self, batch_size):
        states, actions, next_states, rewards, dones = self.buffer.sample(batch_size)
        return states, actions, rewards, next_states, dones
    def add(self, obs, action, reward, done):
        next_obs = obs.copy()
        infos = [{}] * len(obs)
        self.buffer.add(obs,next_obs,action,reward,done,infos)
    def __len__(self):
        return len(self.buffer)

device= 'cuda'
#logger = WandBLogger(entity='jacobhadamczyk', project='test')
mlp = make_mlp(env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.n, hidden_dims=[32, 32], device=device)
agent = SoftQAgent(
    env,
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
    device='cuda',
    learning_starts=512
)
sb3_buff_agent = SoftQAgent(
    env,
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
    device='cuda',
    learning_starts=512
)
sb3_buff_agent.buffer = SB3_Buff_Adapter(ReplayBuffer(
    sb3_buff_agent.buffer.buffer_size,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
))

import time
times = []
for i in range(10):
    start = time.time()
    agent.learn(total_timesteps=5000)
    times.append(time.time() - start)
print(f"Custom buffer time: {np.mean(times)}, std {np.std(times)}",)
times = []
for i in range(10):
    start = time.time()
    sb3_buff_agent.learn(total_timesteps=5000)
    times.append(time.time() - start)
print(f"SB3 buffer time: {np.mean(times)}, std {np.std(times)}")


