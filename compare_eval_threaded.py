import time
import numpy as np
from Architectures import make_atari_nature_cnn, make_mlp
from DQN import DQN
import gymnasium as gym

from Logger import NullLogger

# env = gym.make('CartPole-v1')
n_steps = 10000
# Note: we eliminate all logging and evaluation for this comparison

# sb3_agent = sb3_DQN('MlpPolicy', 
#                     env, 
#                     learning_rate=0.001,
#                     buffer_size=n_steps,
#                     learning_starts=0,
#                     target_update_interval=10,
#                     )
# our_agent = our_DQN(env,
#                     loggers=(NullLogger(),),
#                     architecture = make_mlp,
#                     architecture_kwargs = {'input_dim': env.observation_space.shape[0],
#                                            'output_dim': env.action_space.n,
#                                            'hidden_dims': [64, 64],
#                                            'device': 'cpu'},
#                     learning_rate=0.001,
#                     train_interval=4,
#                     gradient_steps=1,
#                     batch_size=32,
#                     use_target_network=True,
#                     target_update_interval=10,
#                     buffer_size=n_steps,
#                     exploration_fraction=0.1,
#                     log_interval=n_steps,
#                     polyak_tau=1.0,
#                     device='cpu')

def time_learning(agent, n_steps):
    start = time.time()
    agent.learn(n_steps)
    end = time.time()
    return end - start



env = 'ALE/Pong-v5'

threaded_agent = DQN(env,
                    loggers=(NullLogger(),),
                    architecture=make_atari_nature_cnn,
                    architecture_kwargs={'output_dim': gym.make(env).action_space.n},
                    learning_rate=0.001,
                    train_interval=4,
                    gradient_steps=1,
                    batch_size=32,
                    use_target_network=True,
                    target_update_interval=100,
                    buffer_size=n_steps,
                    exploration_fraction=0.1,
                    log_interval=n_steps // 10,
                    polyak_tau=1.0,
                    use_threaded_eval=True)


unthreaded_agent = DQN(env,
                    loggers=(NullLogger(),),
                    architecture=make_atari_nature_cnn,
                    architecture_kwargs={'output_dim': gym.make(env).action_space.n},
                    learning_rate=0.001,
                    train_interval=4,
                    gradient_steps=1,
                    batch_size=32,
                    use_target_network=True,
                    target_update_interval=100,
                    buffer_size=n_steps,
                    exploration_fraction=0.1,
                    log_interval=n_steps // 10,
                    polyak_tau=1.0,
                    use_threaded_eval=False)
                       

unthreaded_time = np.mean([time_learning(unthreaded_agent, n_steps) for _ in range(3)])
threaded_time = np.mean([time_learning(threaded_agent, n_steps) for _ in range(3)])

print(f"Un-threaded (standard) evaluation training took {unthreaded_time:.2f} seconds")
print(f"Threaded (new) evaluation training agent took {threaded_time:.2f} seconds")