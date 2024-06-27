import os
import random

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3.common.logger import configure
import time

import torch
import wandb

def env_id_to_envs(env_id, render):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env


def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        # another run may be creating a folder:
        time.sleep(0.5)
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            # another run may be creating a folder:
            time.sleep(0.5)
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"

        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        logger = configure(format_strings=["stdout"])

    return logger

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

