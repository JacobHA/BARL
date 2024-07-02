import gymnasium as gym
import sys

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import matplotlib.pyplot as plt

sys.path.append('./')
from Logger import TensorboardLogger
from Buffer import Buffer, Buffer2

env = gym.make('LunarLander-v2')
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

# Create a custom buffer and add the data
sb3_buffer = SB3_Buff_Adapter(ReplayBuffer(10_000, env.observation_space, env.action_space))#, optimize_memory_usage=True, handle_timeout_termination=False))
sb3_buffer = Buffer2(10_000, env.observation_space.sample(), env.action_space.sample())

our_buffer = Buffer(10_000, env.observation_space.sample(), env.action_space.sample())

# Add the data:

# Sample from the buffer and compare the results

import time
def test_buffer_sample_speed(buffer, name="Buffer", batch_size=256):
    sample_times = []
    add_times = []
    for i in range(10):
        for j in range(100_000):
            # fill up the buffer
            state = np.array([env.observation_space.sample()])
            action = np.array([env.action_space.sample()])
            reward = np.random.rand(1,1)
            # next_state = np.array([env.observation_space.sample()])
            done = np.random.randint(0, 2, 1)
            start = time.time()
            buffer.add(state, action, reward, done)
            add_times.append(time.time() - start)

        start = time.time()
        # Simulate 500 gradient steps needing buffer samples
        for sample in range(100):
            # Sample from the buffer with batch size 256 and compare results:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        sample_times.append(time.time() - start)
    print(f"{name} time: {np.mean(sample_times)}, std {np.std(sample_times)}",)
    return (np.mean(sample_times), np.std(sample_times)), (np.mean(add_times), np.std(add_times))

sb3s = []
ours = []
bs = []
# Different batch size scaling:
for batch_size in np.arange(2,10):
    batch_size = 2**batch_size
    bs.append(batch_size)

    sb3s.append(test_buffer_sample_speed(sb3_buffer, name="SB3 Buffer", batch_size=batch_size))

    ours.append(test_buffer_sample_speed(our_buffer, name="Our Buffer", batch_size=batch_size))



def plot_times(batch_sizes, sb3_data, our_data, ylabel, title):
    sb3_means = [result[0] for result in sb3_data]
    sb3_stds = [result[1] for result in sb3_data]
    our_means = [result[0] for result in our_data]
    our_stds = [result[1] for result in our_data]

    plt.figure()
    plt.errorbar(batch_sizes, sb3_means, yerr=sb3_stds, label='SB3 Buffer', capsize=5, fmt='-o')
    plt.errorbar(batch_sizes, our_means, yerr=our_stds, label='Our Buffer', capsize=5, fmt='-o')
    plt.xlabel('Batch Size')
    plt.xscale('log', base=2)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'buffer_comparisons_{title.lower().replace(" ", "_")}.png', bbox_inches='tight')

# Plot Sampling Times
sb3_sampling_data = [result[0] for result in sb3s]
our_sampling_data = [result[0] for result in ours]
plot_times(bs, sb3_sampling_data, our_sampling_data, 'Time (seconds)', 'Sampling Times')

# Plot Adding Times
sb3_adding_data = [result[1] for result in sb3s]
our_adding_data = [result[1] for result in ours]
plot_times(bs, sb3_adding_data, our_adding_data, 'Time (seconds)', 'Adding Times')
