
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, seed=None, options=None):
        # this "info" being returned is not quite right (need to stack), but we don't use it anyway
        ob, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))._force()


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
    

class PermuteAtariObs(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = self.env
        new_shape = (self.observation_space.shape[-1], *self.observation_space.shape[:-1])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.transpose([2,0,1]),
            high=self.observation_space.high.transpose([2,0,1]),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )
        self.action_space = self.action_space

    def step(self, *args, **kwargs):
        res = self.env.step(*args, **kwargs)
        newres = (np.transpose(res[0], [2,1,0]), *res[1:])
        del res
        return newres

    def reset(self, *args, **kwargs):
        res, info = self.env.reset(*args, **kwargs)
        res = np.transpose(res, [2,1,0])
        return res, info

# Fire on reset env wrapper:
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs, {}
