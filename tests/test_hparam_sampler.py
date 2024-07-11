import unittest
import random
import numpy as np
from unittest.mock import patch

# Assuming the function `sample_wandb_hyperparams` is defined in a module named `hyperparam_sampling`
# from hyperparam_sampling import sample_wandb_hyperparams
import sys
sys.path.append('./')
from utils import sample_wandb_hyperparams


class TestSampleWandbHyperparams(unittest.TestCase):
    @patch('random.choice')
    @patch('random.uniform')
    @patch('random.normalvariate')
    def test_sample_values(self, mock_normalvariate, mock_uniform, mock_choice):
        mock_choice.side_effect = lambda x: x[0]
        mock_uniform.side_effect = lambda a, b: (a + b) / 2
        mock_normalvariate.side_effect = lambda mean, std: mean

        params = {
            'param1': {'values': [1, 2, 3]},
            'param2': {'distribution': 'uniform', 'min': 0, 'max': 10},
            'param3': {'distribution': 'q_uniform', 'min': 1, 'max': 5},
            'param4': {'distribution': 'normal', 'mean': 0, 'std': 1},
            'param5': {'distribution': 'log_uniform_values', 'min': 1, 'max': 10},
            'param6': {'distribution': 'q_log_uniform_values', 'min': 1, 'max': 10}
        }

        expected = {
            'param1': 1,
            'param2': 5.0,
            'param3': 3,
            'param4': 0,
            'param5': np.exp((np.log(10) + np.log(1)) / 2),
            'param6': int(np.exp((np.log(10) + np.log(1)) / 2))
        }

        result = sample_wandb_hyperparams(params)
        self.assertEqual(result, expected)
        # Check the q distribution params are int:
        self.assertIsInstance(result['param3'], int)
        self.assertIsInstance(result['param6'], int)

    def test_not_implemented_distribution(self):
        params = {
            'param1': {'distribution': 'unknown', 'min': 0, 'max': 10}
        }
        with self.assertRaises(NotImplementedError):
            sample_wandb_hyperparams(params)

    def test_not_implemented_format(self):
        params = {
            'param1': {'min': 0, 'max': 10}
        }
        with self.assertRaises(NotImplementedError):
            sample_wandb_hyperparams(params)

if __name__ == '__main__':
    unittest.main()
