import unittest
import sys
sys.path.append('./')
import gymnasium as gym

from Buffer import Buffer


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        buffer_size = 512
        device = 'cpu'
        self.buffer = Buffer(
            buffer_size=buffer_size,
            state=self.env.observation_space.sample(),
            action=self.env.action_space.sample(),
            device=device
        )

    def test_add(self):
        state, _ = self.env.reset()
        action = self.env.action_space.sample()
        reward = 1
        done = False
        self.buffer.add(state=state, action=action, reward=reward, done=done)
        self.assertEqual(self.buffer.n_stored, 1)

    def test_sample(self):
        batch_size = 32
        state, _ = self.env.reset()
        reward = 1
        done = False
        for i in range(batch_size - 1):
            action = self.env.action_space.sample()
            self.buffer.add(state=state, action=action, reward=reward, done=done)
            state, reward, done, truncated, into = self.env.step(action)
            if done:
                state, _ = self.env.reset()
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)

    def test_multiple_samples(self):
        batch_size = 32
        state, _ = self.env.reset()
        reward = 1
        done = False
        for i in range(5 * batch_size - 1):
            action = self.env.action_space.sample()
            self.buffer.add(state=state, action=action, reward=reward, done=done)
            state, reward, done, truncated, into = self.env.step(action)
            if done:
                state, _ = self.env.reset()
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states2, actions2, rewards2, next_states2, dones2 = self.buffer.sample(batch_size)
        self.assertNotEqual((states[0] - states2[0]).sum(), 0)
        self.assertNotEqual((actions[0] - actions2[0]).sum(), 0)
