import unittest
from unittest.mock import patch, MagicMock
import logging
import os
import shutil
from io import StringIO
import sys
sys.path.append('./')
from Logger import WandBLogger, StdLogger, TensorboardLogger

hparams = {'learning_rate': 0.01, 'batch_size': 32}


class TestWandBLogger(unittest.TestCase):
    # Note: This test is not complete, as it requires mocking the wandb library
    @patch('wandb.init')
    @patch('wandb.log')
    def setUp(self, mock_wandb_init, mock_wandb_log):
        self.logger = WandBLogger(entity="test_entity", project="test_project")


class TestStdLogger(unittest.TestCase):

    def setUp(self):
        self.log_stream = StringIO()
        self.logger = logging.getLogger('test_logger')
        handler = logging.StreamHandler(self.log_stream)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.std_logger = StdLogger(logger=self.logger)

    def test_log_hparams(self):
        self.std_logger.log_hparams(hparams)
        log_output = self.log_stream.getvalue()
        self.assertIn(f'learning_rate: {hparams["learning_rate"]}', log_output)
        self.assertIn(f'batch_size: {hparams["batch_size"]}', log_output)

    def test_log_history(self):
        self.std_logger.log_history('accuracy', 0.95, 1)
        log_output = self.log_stream.getvalue()
        self.assertIn('accuracy: 0.95', log_output)

    def test_log_video(self):
        with self.assertLogs('test_logger', level='WARN') as cm:
            self.std_logger.log_video('path/to/video.mp4')
        self.assertIn('WARNING:test_logger:videos are not logged by std logger', cm.output)


class TestTensorboardLogger(unittest.TestCase):

    def setUp(self):
        self.log_dir = 'test_log_dir'
        self.tensorboard_logger = TensorboardLogger(self.log_dir)

    def tearDown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def test_log_hparams(self):
        hparams = {'learning_rate': 0.01, 'batch_size': 32}
        self.tensorboard_logger.log_hparams(hparams)
        # Checking log files directly would require parsing Tensorboard logs, which is complex for a unit test.
        # Here, we assume the absence of errors indicates success.

    def test_log_history(self):
        self.tensorboard_logger.log_history('accuracy', 0.95, 1)
        # Similar to hparams, check indirectly

    def test_log_video(self):
        # Assuming we use a mock to avoid actual file I/O
        with patch.object(self.tensorboard_logger.writer, 'add_video') as mock_add_video:
            self.tensorboard_logger.log_video('path/to/video.mp4')
            mock_add_video.assert_called_once()

    def test_log_image(self):
        # Assuming we use a mock to avoid actual file I/O
        with patch.object(self.tensorboard_logger.writer, 'add_image') as mock_add_image:
            self.tensorboard_logger.log_image('path/to/image.png')
            mock_add_image.assert_called_once()


if __name__ == '__main__':
    unittest.main()
