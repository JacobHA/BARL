import unittest
import gymnasium as gym
import sys
sys.path.append('./')
from utils import env_id_to_envs
from Architectures import make_atari_nature_cnn
import torch

class TestCNN(unittest.TestCase):
    def test_cnn_with_stacked_states(self):
        # Initialize the environment
        env_id = "ALE/Pong-v5"
        env, eval_env = env_id_to_envs(env_id, render=True, is_atari=True, permute_dims=True)

        # Create the CNN
        cnn = make_atari_nature_cnn(output_dim=env.action_space.n, input_dim=(84, 84, 4))

        # Reset the environment and prepare the state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Check the shape of the CNN output
        output_shape = cnn(state).shape[1]
        expected_shape = env.action_space.n
        self.assertEqual(output_shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
