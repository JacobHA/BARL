import unittest
import torch

from Architectures import make_carracing_cnn


class TestCarRacingCNN(unittest.TestCase):
    def test_carracing_cnn_rgb_observation(self):
        n_actions = 5
        cnn = make_carracing_cnn(output_dim=n_actions, input_dim=(96, 96, 3))

        state = torch.randint(0, 256, (96, 96, 3), dtype=torch.uint8)
        qvals = cnn(state)

        self.assertEqual(tuple(qvals.shape), (1, n_actions))


if __name__ == '__main__':
    unittest.main()
