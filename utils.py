import random
from typing import Optional, Union, Callable
import copy

import gymnasium as gym
import numpy as np

import torch
import wandb

from wrappers import FireResetEnv, FrameStack, PermuteAtariObs

def env_id_to_envs(env_id, render):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env
    

def log_class_vars(self, logger, params, use_wandb=False):
    for item in params:
        value = self.__dict__[item]
        # TODO: Maybe change this logic so that params are always of same type so we don't have to check for tensors here:?
        # TODO: Somehow streamline wandb vs stdout vs tensorboard logging:?
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(item, value)
        if use_wandb:
            wandb.log({item: value})


def sample_wandb_hyperparams(params):
    sampled = {}
    for k, v in params.items():
        if 'values' in v:
            sampled[k] = random.choice(v['values'])
        elif 'distribution' in v:
            if v['distribution'] in {'uniform', 'q_uniform'} or v['distribution'] in {'q_uniform_values', 'uniform_values'}:
                val = random.uniform(v['min'], v['max'])
                if v['distribution'].startswith("q_"):
                    val = int(val)
                sampled[k] = val
            elif v['distribution'] == 'normal':
                sampled[k] = random.normalvariate(v['mean'], v['std'])
            elif v['distribution'] in {'log_uniform_values', 'q_log_uniform_values'}:
                emin, emax = np.log(v['max']), np.log(v['min'])
                sample = np.exp(random.uniform(emin, emax))
                if v['distribution'].startswith("q_"):
                    sample = int(sample)
                sampled[k] = sample
            else:
                raise NotImplementedError(f"Distribution {v['distribution']} not recognized.")
        else:
            raise NotImplementedError(f"{k} format of parameter not recognized: {v}. "
                                      f"Expected a set of values or a distribution")
        assert k in sampled, f"Hparam {k} not successfully sampled."
    return sampled


def find_torch_modules(module, modules=None, prefix=None):
    """
    Recursively find all torch.nn.Modules within a given module.
    Args:
        module (nn.Module): The module to inspect.
        modules (dict, optional): A dictionary to collect module names and their instances.
        prefix (str, optional): A prefix for the module names to handle nested structures.
    Returns:
        dict: A dictionary with module names as keys and module instances as values.
    """
    if modules is None:
        modules = {}
    # Check if the current module itself is an instance of nn.Module
    submodules = None
    if isinstance(module, torch.nn.Module):
        modules[prefix] = module.state_dict()
        submodules = module.named_children
    elif hasattr(module, '__dict__'):
        submodules = module.__dict__.items
    # Recursively find all submodules if the current module is a container
    if submodules:
        for name, sub_module in submodules():
            if prefix:
                mod_name = f"{prefix}.{name}"
            else:
                mod_name = name
            if name in mod_name.split('.'):
                continue
            find_torch_modules(sub_module, modules, mod_name)

    return modules

def polyak(target_nets, online_nets, tau):
    """
    Perform a Polyak (exponential moving average) update for target networks.
    
    Uses the standard definition: target = (1 - tau) * target + tau * online
    
    Args:
        target_nets: Target network whose parameters will be updated.
        online_nets: Online network whose parameters will be used for the update.
        tau (float): The soft update coefficient, in range [0, 1].
                     tau=0: no update (target unchanged)
                     tau=1: hard update (target = online)
    """
    with torch.no_grad():
        for new_params, target_params in zip(online_nets.parameters(), target_nets.parameters()):
            for param, target_param in zip(new_params, target_params):
                # Standard Definition: Target = (1-tau)*Target + tau*Online
                target_param.data.mul_(1.0 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

def prepare_online_and_target(use_target_network, architecture, architecture_kwargs):
    # Initialize online net(s):
    online_nets = architecture(**architecture_kwargs)
    if use_target_network:
        # Make another instance of the architecture for the target network:
        target_nets = architecture(**architecture_kwargs)
        target_nets.load_state_dict(online_nets.state_dict())
    else:
        target_nets = online_nets
    return online_nets, target_nets 

def check_polyak_tau(use_target_network, polyak_tau, target_update_interval):
    if use_target_network:
        if polyak_tau is not None:
            if not(0 <= polyak_tau <= 1):
                raise ValueError("Polyak tau must be in the range [0, 1].")
        else:
            raise ValueError("WARNING: No polyak tau specified for soft target updates. Use tau=1 for hard updates.")

        if target_update_interval is None:
            raise ValueError("ERROR: Target network update interval not specified.")
        
def auto_device(device: Union[torch.device, str] = 'auto'):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return device
    
def zip_strict(*iterables):
    """
    zip() function but enforces that iterables are of equal length.
    Raises ValueError if iterables are not of equal length.

    :param *iterables: iterables to zip()
    """

    # Yield the zipped items
    yield from zip(*iterables, strict=True)

def env_id_to_envs(env_id, render, is_atari=False, permute_dims=False):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    if is_atari:
        return atari_env_id_to_envs(env_id, render, n_envs=1, frameskip=4, framestack_k=4, permute_dims=permute_dims)
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env

from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

def atari_env_id_to_envs(env_id, render, n_envs, frameskip=1, framestack_k=None, grayscale_obs=True, permute_dims=False):
    if isinstance(env_id, str):
        # Don't vectorize if there is only one env
        if n_envs==1:
            env = gym.make(env_id, frameskip=frameskip)
            env = AtariPreprocessing(env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                env = FrameStack(env, framestack_k)
            # permute dims for nature CNN in sb3
            if permute_dims:
                env = PermuteAtariObs(env)

            # make another instance for evaluation purposes only:
            eval_env = gym.make(env_id, render_mode='human' if render else None, frameskip=frameskip)
            eval_env = AtariPreprocessing(eval_env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                eval_env = FrameStack(eval_env, framestack_k)
            if permute_dims:
                eval_env = PermuteAtariObs(eval_env)

            # if render:
            #     eval_env = RecordVideo(eval_env, video_folder='videos')
            env = FireResetEnv(env)
            eval_env = FireResetEnv(eval_env)
        else:
            env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

            eval_env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

    elif isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        eval_env = copy.deepcopy(env_id)
    else:
        env = env_id
        # Make a new copy for the eval env:
        eval_env = copy.deepcopy(env_id)

    return env, eval_env


def make_video_env(
    env_id,
    video_folder: str,
    name_prefix: str,
    is_atari: bool = False,
    permute_dims: bool = False,
    frameskip: int = 4,
    framestack_k: Optional[int] = 4,
    grayscale_obs: bool = True,
    episode_trigger: Optional[Callable[[int], bool]] = None,
):
    """
    Create a single evaluation env configured for video recording.
    Uses render_mode="rgb_array" to ensure video capture works across environments.
    """
    try:
        from gymnasium.wrappers import RecordVideo
    except Exception:
        RecordVideo = None

    if isinstance(env_id, gym.Env):
        eval_env = copy.deepcopy(env_id)
    elif is_atari:
        eval_env = gym.make(env_id, render_mode='rgb_array', frameskip=frameskip)
        eval_env = AtariPreprocessing(
            eval_env,
            terminal_on_life_loss=True,
            screen_size=84,
            grayscale_obs=grayscale_obs,
            grayscale_newaxis=True,
            scale_obs=False,
            noop_max=30,
            frame_skip=1,
        )
        if framestack_k:
            eval_env = FrameStack(eval_env, framestack_k)
        if permute_dims:
            eval_env = PermuteAtariObs(eval_env)
        eval_env = FireResetEnv(eval_env)
    else:
        eval_env = gym.make(env_id, render_mode='rgb_array')

    if RecordVideo is not None and video_folder:
        if episode_trigger is None:
            episode_trigger = lambda episode_id: True
        eval_env = RecordVideo(
            eval_env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=episode_trigger,
            disable_logger=True,
        )

    return eval_env
