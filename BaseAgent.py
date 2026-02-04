import os
import time
import numpy as np
import torch
import json
import gymnasium as gym
from typing import Optional, Union, Tuple, List
import tqdm
from typeguard import typechecked
from utils import auto_device, env_id_to_envs, find_torch_modules
from Logger import BaseLogger, StdLogger
from Buffer import Buffer
from video_utils import VideoRecorder, resolve_video_dir
from network_monitor import EmptyMonitor
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
                 architecture: Union[str, torch.nn.Module, callable] = "mlp",
                 learning_rate: float = 3e-4,
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
                 eval_callbacks: List[callable] = [],
                 record_eval_video: bool = False,
                 eval_video_every: int = 1,
                 eval_video_episodes: int = 1,
                 network_monitor: Optional[callable] = None,
                 ) -> None:

        self.LOG_PARAMS = {
            'train/env. steps': 'env_steps',
            'eval/avg_reward': 'avg_eval_rwd',
            'eval/auc': 'eval_auc',
            'train/num. episodes': 'num_episodes',
            'train/fps': 'train_fps',
            'train/num. updates': '_n_updates',
            'train/lr': 'learning_rate',
        }
        self.learn_env_steps = 0
        self.total_env_steps = 0
        self.total_learn_env_steps = 0
        self.kwargs = get_new_params(None, locals())
        is_atari = False
        permute_dims = False
        if isinstance(env_id, str):
            if 'ALE' in env_id or 'NoFrameskip' in env_id:
                is_atari=True
                permute_dims=True
            self.env_str = env_id
        self.env_id = env_id
        self.is_atari = is_atari
        self.permute_dims = permute_dims
        self.env, self.eval_env = env_id_to_envs(env_id, render, is_atari=is_atari, permute_dims=permute_dims)

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
        self.algo_name = "BaseAgent"  # Override in subclasses

        self.gradient_steps = gradient_steps
        self.device = auto_device(device)

        #TODO: Implement save_checkpoints
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval

        self.train_interval = train_interval
        if isinstance(train_interval, tuple):
            raise NotImplementedError("train_interval as a tuple is not supported yet.\
                                       \nEnter int corresponding to env_steps")
        self.max_grad_norm = max_grad_norm
        self.learning_starts = learning_starts
        self.eval_callbacks = [eb(self) for eb in eval_callbacks]
        self.avg_eval_rwd = None
        self.fps = None
        self.train_this_step = False

        self.buffer = Buffer(
            buffer_size=buffer_size,
            state=self.env.observation_space.sample(),
            action=self.env.action_space.sample(),
            device=device
        )

        self.eval_auc = 0
        self.num_episodes = 0

        self.video_recorder = VideoRecorder(
            record_enabled=record_eval_video,
            record_every=eval_video_every,
            num_episodes=eval_video_episodes,
        )
        
        self.network_monitor = network_monitor if network_monitor is not None else EmptyMonitor()
        self._n_updates = 0

    def log_hparams(self, hparam_dict):
        for logger in self.loggers:
            logger.log_hparams(hparam_dict)

    def log_history(self, param, val, step):
        for logger in self.loggers:
            logger.log_history(param, val, step)

    def exploration_policy(self, state: np.ndarray):
        raise NotImplementedError()
    
    def evaluation_policy(self, state: np.ndarray):
        raise NotImplementedError()

    
    def gradient_step(self, grad_step: int) -> None:
        """
        Perform a single gradient step on the agent's networks
        """
        raise NotImplementedError()

    def _train(self, gradient_steps) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        # Increase update counter
        self._n_updates += gradient_steps
        for grad_step in range(gradient_steps):
            self.gradient_step(grad_step)
        
        # Monitor networks after updates
        self.network_monitor(self)
        
    def learn(self, total_timesteps: int):
        """
        Train the agent for total_timesteps
        """
        # Write running status at start of training
        self._write_status('running')
        try:
            if self.network_monitor is not None and not hasattr(self, "monitor_samples"):
                self._initialize_monitor_samples(n_samples=1000)
            # Start a timer to log fps:
            init_train_time = time.thread_time_ns()
            self.learn_env_steps = 0
            self.total_learn_env_steps = total_timesteps
            with tqdm.tqdm(total=total_timesteps, desc="Training") as pbar:

                while self.learn_env_steps < total_timesteps:
                    state, _ = self.env.reset()

                    done = False
                    self.num_episodes += 1
                    self.rollout_reward = 0
                    avg_ep_len = 0
                    while not done and self.learn_env_steps < total_timesteps:
                        action = self.exploration_policy(state)

                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        self._on_step()
                        avg_ep_len += 1
                        done = terminated or truncated
                        self.rollout_reward += reward

                        self.train_this_step = (self.train_interval == -1 and terminated) or \
                            (self.train_interval != -1 and self.learn_env_steps %
                            self.train_interval == 0)

                        # Add the transition to the replay buffer:
                        action = np.array([action])
                        state_array = np.asarray(state)
                        self.buffer.add(state_array, action, reward, terminated)
                        state = next_state
                        if self.learn_env_steps % self.log_interval == 0:
                            train_time = (time.thread_time_ns() - init_train_time) / 1e9
                            train_fps = self.log_interval / train_time
                            self.log_history('time/train_fps', train_fps, self.learn_env_steps)
                            self.avg_eval_rwd = self.evaluate()
                            # Log buffer statistics
                            buffer_stats = self.buffer.calculate_statistics()
                            self.log_history('buffer/n_stored', buffer_stats['n_stored'], self.learn_env_steps)
                            self.log_history('buffer/terminated_fraction', buffer_stats['terminated_fraction'], self.learn_env_steps)
                            # Save reward histogram to a separate file
                            # TODO: make this part of the logger class
                            for logger in self.loggers:
                                if logger.run_dir is not None:
                                    histogram_path = os.path.join(logger.run_dir, 'reward_histogram.json')
                                    with open(histogram_path, 'w') as f:
                                        json.dump(buffer_stats['reward_histogram'], f)
                                    break
                            init_train_time = time.thread_time_ns()
                            pbar.update(self.log_interval)

                    if done:
                        self.log_history("rollout/ep_reward", self.rollout_reward, self.learn_env_steps)
                        self.log_history("rollout/avg_episode_length", avg_ep_len, self.learn_env_steps)
                        self.log_history("train/num. episodes", self.num_episodes, self.learn_env_steps)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Cleaning up...")
        finally:
            # Cleanup after training completes or is interrupted
            self._cleanup()
 
    def _initialize_monitor_samples(self, n_samples: int = 1000) -> None:
        """Sample fixed random states/actions for network monitoring."""
        states = [self.env.observation_space.sample() for _ in range(n_samples)]
        actions = [self.env.action_space.sample() for _ in range(n_samples)]
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)


        self.monitor_samples = {"states": states, "actions": actions}

    def _get_run_dir(self) -> Optional[str]:
        """Get the run directory from loggers."""
        for logger in self.loggers:
            if hasattr(logger, 'run_dir') and logger.run_dir:
                return logger.run_dir
        return None
    
    def _write_status(self, status: str) -> None:
        """Write run_data.json with status in run directory."""
        run_dir = self._get_run_dir()
        if run_dir:
            self._write_run_data(status)
    
    def _write_run_data(self, status: str) -> None:
        """Write run_data.json with metadata for dashboard organization."""
        run_dir = self._get_run_dir()
        if run_dir:
            run_data_path = os.path.join(run_dir, 'run_data.json')
            named_networks = []
            monitor = self.network_monitor
            if monitor is not None and hasattr(monitor, 'networks_to_monitor'):
                networks_to_monitor = monitor.networks_to_monitor
                if isinstance(networks_to_monitor, dict):
                    named_networks = list(networks_to_monitor.keys())
                elif isinstance(networks_to_monitor, (list, tuple, set)):
                    named_networks = list(networks_to_monitor)
            run_data = {
                'algo_name': self.algo_name,
                'env_str': self.env_str,
                'status': status,
                'named_networks': named_networks,
            }
            try:
                with open(run_data_path, 'w') as f:
                    json.dump(run_data, f)
            except Exception as e:
                print(f"Warning: Failed to write run_data.json: {e}")
    
    def _cleanup(self) -> None:
        """
        Cleanup method called after training completes.
        Closes loggers and cleans up resources.
        """
        # Write appropriate status before cleanup
        self._write_status('stopped')
        
        # Terminate any background preloading processes in the buffer
        if hasattr(self, 'buffer'):
            self.buffer.cleanup()
        
        # Close all loggers
        for logger in self.loggers:
            logger.close()
    
    def _on_step(self) -> None:
        """
        This method is called after every step in the environment
        """
        self.learn_env_steps += 1
        self.total_env_steps += 1

        if self.train_this_step:
            if self.learn_env_steps > self.learning_starts:
                self._train(self.gradient_steps)

    def _log_stats(self):
        # Get the current learning rate from the optimizer:
        for log_name, class_var in self.LOG_PARAMS.items():
            self.log_history(log_name, self.__dict__[class_var], self.learn_env_steps)

    def evaluate(self, n_episodes=10) -> float:
        # run the current policy and return the average reward
        avg_reward = 0.0
        n_steps = 0
        init_eval_time = time.process_time_ns()
        
        # Video recording setup
        record_this_eval = self.video_recorder.should_record()
        eval_env = self.eval_env
        video_env_created = False
        video_name_prefix = None
        video_episode_rewards = []
        video_dir = None

        if record_this_eval:
            video_dir = resolve_video_dir(self.loggers)
            if video_dir:
                eval_env, video_name_prefix = self.video_recorder.get_video_env(
                    self.env_id,
                    video_dir,
                    self.learn_env_steps,
                    self.is_atari,
                    self.permute_dims,
                )
                video_env_created = True

        try:
            for ep in range(n_episodes):
                state, _ = eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action = self.evaluation_policy(state)
                    n_steps += 1
                    next_state, reward, terminated, truncated, info = eval_env.step(
                        action)
                    state = next_state
                    avg_reward += reward
                    ep_reward += reward
                    done = terminated or truncated
                    for callback in self.eval_callbacks:
                        callback(state=state, action=action, reward=reward, done=done, end=False)
                if record_this_eval and ep < self.video_recorder.num_episodes:
                    video_episode_rewards.append(ep_reward)
        finally:
            if video_env_created:
                try:
                    eval_env.close()
                except Exception:
                    # Intentionally ignore errors during environment cleanup to avoid
                    # masking earlier exceptions or breaking the evaluation flow.
                    pass
        eval_time = (time.process_time_ns() - init_eval_time) / 1e9
        avg_reward /= n_episodes
        eval_fps = n_steps / eval_time
        
        self.log_history('eval/avg_reward', avg_reward, self.learn_env_steps)
        self.log_history('eval/avg_episode_length', n_steps / n_episodes, self.learn_env_steps)
        self.log_history('eval/time', eval_time, self.learn_env_steps)
        self.log_history('eval/fps', eval_fps, self.learn_env_steps)
        if video_episode_rewards:
            video_avg_reward = float(np.mean(video_episode_rewards))
            self.log_history('eval/video_avg_reward', video_avg_reward, self.learn_env_steps)
            if video_name_prefix and video_dir:
                self.video_recorder._save_video_metadata(
                    video_dir, video_name_prefix, video_avg_reward, self.learn_env_steps
                )
        for callback in self.eval_callbacks:
            callback(self, end=True)
        return avg_reward

    def save(self, path=None):
        if path is None:
            path = str(self)
        # save the number of time steps:
        self.kwargs['num_timesteps'] = self.learn_env_steps
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
    wandb_logger = WandBLogger(entity="jacobhadamczyk", project="test")
    agent = BaseAgent("CartPole-v1", loggers=(wandb_logger,))
