from typing import Optional, Union, Tuple
import threading
import multiprocessing as mp

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
            preload_sample:bool=True,
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

        self.transforms = {}
        self.clear()
        self.done_handlers = done_handlers
        self.preload_sample = preload_sample
        self.preloaded_sample = None
        self.preload_done = False
        self.proc = None
        self.n_missed_preloads = 0

    def _preload(self, batch_size, queue):
        preloaded_sample = self._sample(batch_size)
        queue.put(preloaded_sample)

    def preload(self, batch_size):
        def preload_worker():
            queue = mp.Queue()
            self.proc = mp.Process(target=self._preload, args=(batch_size,queue))
            self.proc.start()
            # self._preload(batch_size, queue)
            self.preloaded_sample = queue.get()
            self.preload_done = True
            # print("Preloading done")
            self.proc.join()
            self.proc = None
            # print("joined")
        if self.proc is not None:
            # print(f"previous preloading still running {self.n_missed_preloads}")
            self.n_missed_preloads += 1
        else:
            self.n_missed_preloads = 0
            worker = threading.Thread(target=preload_worker)
            worker.start()

    def clear(self):
        self.states =  np.empty((self.buffer_size, *self.state_shape),  dtype=self.state_dtype)
        self.actions = np.empty((self.buffer_size, *self.action_shape), dtype=self.action_dtype)
        self.rewards = np.empty((self.buffer_size, 1), dtype=np.float32)
        self.dones =   np.empty((self.buffer_size, 1), dtype=bool)
        self.ep_start = 0
        self.ep_end = 0
        self.n_stored = 0

    def add(self, state, action, reward, done):
        self.states [self.ep_end] = state
        self.actions[self.ep_end] = action
        self.rewards[self.ep_end] = reward
        self.dones  [self.ep_end] = done
        self.n_stored = min(self.buffer_size, self.n_stored + 1)
        self.ep_end += 1
        if self.ep_end == self.buffer_size:
            self.ep_end = 0
        if done:
            self._handle_done()

    @staticmethod
    def to_device(batch, device):
        return [th.from_numpy(x).to(device) for x in batch]

    def _sample(self, batch_size):
        # since the s' is not valid where s is done, we need to use
        idxs_done = np.where(self.dones)[0]
        idxs_all = np.arange(self.n_stored)
        idxs_valid = np.setdiff1d(idxs_all, idxs_done)
        idx = np.random.choice(idxs_valid, batch_size)
        return (self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.states[idx + 1],
                self.dones[idx],)

    def sample(self, batch_size):
        if self.preload_done:
            # get the preloaded sample and start preloading the next one
            self.preload_done = False
            self.preload(batch_size)
            return Buffer.to_device(self.preloaded_sample, self.device)
        else:
            # print('preloading not ready, preparing for the next time')
            self.preload(batch_size)
        return Buffer.to_device(self._sample(batch_size), self.device)

    def _handle_done(self):
        for handler, h_kwargs in self.done_handlers:
            handler(self, **h_kwargs)


class TDBuffer(Buffer):
    def __init__(self, *args,  td_steps:int=1 ,**kwargs):
        super().__init__(*args, **kwargs)
        """
        :param td_steps: int - number of steps to look ahead for TD learning
        """
        self.td_steps = td_steps

    def sample(self, batch_size, preloading=False):
        if not preloading and self.preloaded_sample is not None:
            return self.preloaded_sample
        if self.preload_sample and not preloading:
            print('preloading not ready, preparing for the next time')
            self.preloaded_sample = None
            self._preload(batch_size)
        idx = np.random.randint(low=0,high=self.n_stored, size=(batch_size,))