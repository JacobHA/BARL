import json
import os
from typing import Optional, Union, Tuple, List, Callable, Any
from pathlib import Path

import h5py
import numpy as np
import torch as th
from typeguard import typechecked


class DoneHandler:
    @typechecked
    def handle_done(self, buffer: 'Buffer'):
        raise NotImplementedError()

# another example done handler
class RewToGoDoneHandler(DoneHandler):
    def __init__(self, gamma):
        self.gamma = gamma

    @typechecked
    def handle_done(self, buffer: 'Buffer'):
        """turn last episode rewards into reward-to-go"""
        if buffer.ep_start == buffer.ep_end:
            return
        # calculate reward-to-go
        for i in range(buffer.ep_end-1, buffer.ep_start-1, -1):
            buffer.rewards[i] += self.gamma * buffer.rewards[i+1]
        buffer.ep_start = buffer.ep_end


class Buffer:
    """Buffer for storing potentially huge amount of images"""
    @typechecked
    def __init__(
            self,
            buffer_size: int = 20000,
            state:Union[th.Tensor,np.ndarray]=None,
            action:Union[th.Tensor,np.ndarray,int,np.int64]=None,
            done_handlers:Tuple[DoneHandler, ...] = (),
            device:str='cpu',
    ):
        """
        n_samples: int - number of samples to store
        state: th.Tensor or np.ndarray - example state to be stored
        action: th.Tensor or np.ndarray - example action to be stored
        max_in_memory: int - maximum number of samples to store in memory
        img_history_len: int - number of images to store for each sample
        done_handlers: List[Callable] - list of functions to call when an episode is done
        """
        assert isinstance(state, th.Tensor) or isinstance(state, np.ndarray)
        self.state_shape = state.shape
        self.action_shape = (action,) if isinstance(action, int) else action.shape
        self.buffer_size = buffer_size
        self.n_stored = 0
        self.device = device
        self.states = None
        self.actions = None
        self.dones = None
        self.rewards = None
        self.ep_start = None
        self.ep_end = None
        self.transforms = {}
        self.clear()
        self.done_handlers = done_handlers

    def clear(self):
        self.states =  th.empty((self.buffer_size, *self.state_shape), device=self.device)
        self.actions = th.empty((self.buffer_size, *self.action_shape), device=self.device)
        self.rewards = th.empty((self.buffer_size, 1), device=self.device)
        self.dones =   th.empty((self.buffer_size, 1), dtype=bool, device=self.device)
        self.ep_start = 0
        self.ep_end = 0
        self.n_stored = 0

    def add(self, state, action, reward, done):
        self.states[self.ep_end] =  th.tensor(state, device=self.device)
        self.actions[self.ep_end] = th.tensor(action, device=self.device)
        self.rewards[self.ep_end] = th.tensor(reward, device=self.device)
        self.dones[self.ep_end] =   th.tensor(done, device=self.device)
        self.n_stored = min(self.buffer_size, self.n_stored + 1)
        self.ep_end += 1
        if self.ep_end == self.buffer_size:
            self.ep_end = 0
        if done:
            self._handle_done()

    # todo: eager shuffle and load
    def sample(self, batch_size):
        idx = th.randint(high=self.n_stored, size=(batch_size,), device=self.device)
        # todo: valid next state indexing
        # todo: td sampling
        return  self.states[idx], self.actions[idx], self.rewards[idx], self.states[idx + 1], self.dones[idx],

    def _handle_done(self):
        for handler, h_kwargs in self.done_handlers:
            handler(self, **h_kwargs)