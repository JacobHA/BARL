import time
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from typing import Optional, Union, Tuple, List
from typeguard import typechecked
from utils import log_class_vars, env_id_to_envs
from Logger import BaseLogger, StdLogger

# use get_type_hints to throw errors if the user passes in an invalid type:


def get_new_params(base_cls_obj, locals):
    for it in {'self', 'args', 'TypeCheckMemo', 'memo', 'check_argument_types', 'env', 'eval_env', 'architecture'}:
        locals.pop(it, None)
    base_class_kwargs = {} if base_cls_obj is None else base_cls_obj.kwargs

    return {**locals, **base_class_kwargs}


class BaseAgent:
    @typechecked
    def __init__(self,
                 env_id: Union[str, gym.Env],
                 architecture: Union[str, torch.nn.Module] = "mlp",
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 buffer_size: int = 100_000,
                 gradient_steps: int = 1,
                 train_interval: int = 1,
                 max_grad_norm: float = 10,
                 learning_starts=5_000,
                 device: Union[torch.device, str] = "auto",
                 render: bool = False,
                 loggers: Tuple[BaseLogger] = (StdLogger(),),
                 log_interval: int = 1_000,
                 save_checkpoints: bool = False,
                 seed: Optional[int] = None,
                 ) -> None:
        
        self.HPARAM_ATTRS = {
            'learning_rate',
            'batch_size',
            'buffer_size',
            'gradient_steps',
            'train_interval',
            'max_grad_norm',
            'learning_starts',
        }

        self.LOG_PARAMS = {
            'train/env. steps': 'env_steps',
            'eval/avg_reward': 'avg_eval_rwd',
            'eval/auc': 'eval_auc',
            'train/num. episodes': 'num_episodes',
            'train/fps': 'train_fps',
            'train/num. updates': '_n_updates',
            'train/lr': 'learning_rate',
        }


        self.kwargs = get_new_params(None, locals())
        self.env, self.eval_env = env_id_to_envs(env_id, render)

        if hasattr(self.env.unwrapped.spec, 'id'):
            self.env_str = self.env.unwrapped.spec.id
        elif hasattr(self.env.unwrapped, 'id'):
            self.env_str = self.env.unwrapped.id
        else:
            self.env_str = str(self.env.unwrapped)

        self.architecture = architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.loggers = loggers

        self.gradient_steps = gradient_steps
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        #TODO: Implement save_checkpoints
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval

        self.train_interval = train_interval
        if isinstance(train_interval, tuple):
            raise NotImplementedError("train_interval as a tuple is not supported yet.\
                                       \nEnter int corresponding to env_steps")
        self.max_grad_norm = max_grad_norm
        self.learning_starts = learning_starts
        self.avg_eval_rwd = None
        self.fps = None
        self.train_this_step = False
        

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            n_envs=1,
            handle_timeout_termination=True,
            device=device
        )

        self.eval_auc = 0
        self.num_episodes = 0

        self._n_updates = 0
        self.env_steps = 0

    def log_hparams(self, hparam_dict):
        # Log the agent's hyperparameters:
        for logger in self.loggers:
            # hparam_values = {attr: self.__dict__[attr] for attr in self.HPARAM_ATTRS}
            logger.log_hparams(hparam_dict)

    def log_h(self, param, val):
        for logger in self.loggers:
            # hparam_values = {attr: self.__dict__[attr] for attr in self.HPARAM_ATTRS}
            logger.log_history(param, val)

    def _initialize_networks(self):
        raise NotImplementedError()

    def exploration_policy(self, state: np.ndarray):
        raise NotImplementedError()
    
    def evaluation_policy(self, state: np.ndarray):
        raise NotImplementedError()

    def gradient_descent(self, batch):
        raise NotImplementedError()

    def _train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        
        # Increase update counter
        self._n_updates += gradient_steps
        for _ in range(gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.buffer.sample(batch_size)

            loss = self.gradient_descent(batch)
            self.optimizer.zero_grad()

            # Clip gradient norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()


    def learn(self, total_timesteps: int):
        """
        Train the agent for total_timesteps
        """
        # Start a timer to log fps:
        self.initial_time = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()

            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            avg_ep_len = 0
            while not done and self.env_steps < total_timesteps:
                action = self.exploration_policy(state)

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                self._on_step()
                avg_ep_len += 1
                done = terminated or truncated
                self.rollout_reward += reward

                self.train_this_step = (self.train_interval == -1 and terminated) or \
                    (self.train_interval != -1 and self.env_steps %
                     self.train_interval == 0)

                # Add the transition to the replay buffer:
                action = np.array([action])
                state = np.array([state])
                next_state = np.array([next_state])
                sarsa = (state, next_state, action, reward, terminated)
                self.buffer.add(*sarsa, [infos])
                state = next_state
                if self.env_steps % self.log_interval == 0:
                    self._log_stats()

            if done:
                for logger in self.loggers:
                    logger.log_history("rollout/ep_reward", self.rollout_reward)
                    logger.log_history("rollout/avg_episode_length", avg_ep_len)
                

    def _on_step(self):
        """
        This method is called after every step in the environment
        """
        self.env_steps += 1

        if self.train_this_step:
            if self.env_steps > self.learning_starts:
                self._train(self.gradient_steps, self.batch_size)
            

    def _log_stats(self):
        # end timer:
        t_final = time.thread_time_ns()
        # fps averaged over log_interval steps:
        self.train_time = (t_final - self.initial_time) / 1e9
        self.train_fps = self.log_interval / self.train_time

        if self.env_steps > 0:
            self.avg_eval_rwd = self.evaluate()
            self.eval_auc += self.avg_eval_rwd
        
        # Get the current learning rate from the optimizer:
        # log_class_vars(self, self.logger, LOG_PARAMS)
        for log_name, class_var in self.LOG_PARAMS.items():
            for logger in self.loggers:
                logger.log_history(log_name, self.__dict__[class_var])

                # logger.dump(step=self.env_steps)

        self.initial_time = time.thread_time_ns()

    def evaluate(self, n_episodes=10) -> float:
        # run the current policy and return the average reward
        self.initial_time = time.process_time_ns()
        avg_reward = 0.
        n_steps = 0
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.evaluation_policy(state)
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time) / 1e9
        eval_fps = n_steps / eval_time
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        for logger in self.loggers:
            logger.log_history('eval/avg_episode_length', n_steps / n_episodes)
            logger.log_history('eval/time', eval_time)
            logger.log_history('eval/fps', eval_fps)
        return avg_reward

    def save(self, path=None):
        if path is None:
            path = str(self)
        # save the number of time steps:
        self.kwargs['num_timesteps'] = self.env_steps
        self.kwargs['continue_training'] = True
        total_state = {
            "kwargs": self.kwargs,
            "state_dicts": find_torch_modules(self),
            "class": self.__class__.__name__
        }
        # if the path is a directory, make :
        if '/' in path:
            bp = path.split('/')
            base_path = os.path.join(*bp[:-1])
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        torch.save(total_state, path)

    @staticmethod
    def load(path, **new_kwargs):
        state = torch.load(path)
        cls = BaseAgent
        for cls_ in BaseAgent.__subclasses__():
            if cls_.__name__ == state['class']:
                cls = cls_
        args = state['kwargs'].get('args', ())
        kwargs = state['kwargs']
        kwargs.update(new_kwargs)
        agent = cls(*args, **kwargs)
        for k, v in state['state_dicts'].items():
            attrs = k.split('.')
            module = agent
            for attr in attrs:
                module = getattr(module, attr)
            module.load_state_dict(v)
        return agent
    

if __name__ == '__main__':
    from Logger import WandBLogger
    logger = WandBLogger(entity="jacobhadamczyk", project="test")
    agent = BaseAgent("CartPole-v1")
