import unittest
import sys
sys.path.append('./')
from utils import env_id_to_envs
from Architectures import make_mlp

class TestMLP(unittest.TestCase):
    def test_cnn_with_stacked_states(self):
        # Initialize the environment
        env_id = "CartPole-v1"
        env, eval_env = env_id_to_envs(env_id, render=True, is_atari=False, permute_dims=False)

        # Create the CNN
        mlp = make_mlp(input_dim=env.observation_space.shape[0],
                        output_dim=env.action_space.n, 
                        hidden_dims=(64,16,8))

        # Reset the environment and prepare the state
        state, _ = env.reset()

        # Check the shape of the CNN output
        output_shape = mlp(state).shape[0]
        expected_shape = env.action_space.n
        self.assertEqual(output_shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
