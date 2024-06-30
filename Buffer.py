from typing import Optional, Union, Tuple
import threading

import numpy as np
import torch as th
from typeguard import typechecked

from utils import auto_device


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


type_to_th_type = {
    int: th.int64,
    np.int64: th.int64,
}
type_to_np_type = {
    int: np.int64,
}


class Buffer:
    """Buffer for storing potentially huge amount of images"""
    @typechecked
    def __init__(
            self,
            buffer_size: int = 20000,
            state:Union[th.Tensor,np.ndarray]=None,
            action:Union[th.Tensor,np.ndarray,int,np.int64]=None,
            done_handlers:Tuple[DoneHandler, ...] = (),
            device:str='auto',
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
        self.state_dtype = state.dtype if isinstance(state, np.ndarray) else state.dtype
        self.action_shape = (1,) if isinstance(action, (int, np.integer))  else action.shape
        self.action_dtype = action.dtype if isinstance(action, np.ndarray) else type(action)
        self.buffer_size = buffer_size
        self.n_stored = 0
        self.device = auto_device(device)
        self.states = None
        self.actions = None
        self.dones = None
        self.rewards = None
        self.ep_start = None
        self.ep_end = None
        self.transforms = {}
        self.clear()
        self.done_handlers = done_handlers
        self.preload_sample = True
        self.preloaded_sample = None

    def _preload(self, batch_size):
        self.preloaded_sample = self.sample(batch_size, preloading=True)

    def preload(self, batch_size):
        worker = threading.Thread(target=self._preload, args=(batch_size,))
        worker.start()

    def clear(self):
        self.sars =    np.empty((self.buffer_size, 5), dtype=object)
        self.ep_start = 0
        self.ep_end = 0
        self.n_stored = 0

    def add(self, state, action, reward, next_state, done):
        self.sars[self.ep_end] = (state, action, reward, next_state, done)
        self.n_stored = min(self.buffer_size, self.n_stored + 1)
        self.ep_end += 1
        if self.ep_end == self.buffer_size:
            self.ep_end = 0
        if done:
            self._handle_done()

    def sample(self, batch_size, preloading=False):
        if not preloading and self.preloaded_sample is not None:
            # print('Preloaded sample')
            return self.preloaded_sample
        if self.preload_sample and not preloading:
            print('preloading not ready, preparing for the next time')
            self.preloaded_sample = None
            self._preload(batch_size)
        idx = np.random.randint(low=0,high=self.n_stored, size=(batch_size,))
        # Unpack sampled experiences
        batch = [self.sars[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to device
        states = th.from_numpy(np.array([state for state in states], dtype=self.state_dtype)).to(self.device)
        actions = th.from_numpy(np.array([action for action in actions], dtype=self.action_dtype)).to(self.device)
        rewards = th.from_numpy(np.array([reward for reward in rewards], dtype=np.float32)).to(self.device)
        next_states = th.from_numpy(np.array([next_state for next_state in next_states], dtype=self.state_dtype)).to(self.device)
        dones = th.from_numpy(np.array([done for done in dones], dtype=bool)).to(self.device)

        return states, actions, rewards, next_states, dones
    
    def _handle_done(self):
        for handler, h_kwargs in self.done_handlers:
            handler(self, **h_kwargs)