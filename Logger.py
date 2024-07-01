# Wandb, tensorboard, stdout, python logger
from functools import lru_cache
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb


logger_types = {'wandb', 'std', 'tensorboard'}


class BaseLogger: 
    def log_hparams(self, hparam_dict):
        raise NotImplementedError()
    def log_history(self, param, value, step):
        raise NotImplementedError()
    def log_video(self, video_path):
        raise NotImplementedError()
    def log_image(self, image_path):
        raise NotImplementedError()
        

class WandBLogger(BaseLogger):
    def __init__(self, entity, project):
        wandb.init(entity=entity, project=project)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            # check if not serializable:
            try:
                wandb.log({param: value})
            except Exception as e:
                print(f"Could not log {param}: {value}")
    def log_history(self, param, value, step):
        wandb.log({param: value}, step=step)
    def log_video(self, video_path, name="video"):
        wandb.log({name: wandb.Video(video_path)})
    def log_image(self, image_path, name="image"):
        wandb.log({name: wandb.Image(image_path)})


class StdLogger(BaseLogger):
    def __init__(self, logger=None):
        if logger is not None:
            self.log = logger
        else:
            self.log = logging.getLogger("Barl")
            self.log.setLevel(logging.INFO)
            st = logging.StreamHandler()
            st.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            st.setFormatter(formatter)
            self.log.addHandler(st)
            # self.log.setFormatter(formatter)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            # self.log.info(param, value)
            self.log.info(f"{param}: {value}")
    def log_history(self, param, value, step):
        self.log.info(f"{param}: {value}")
    @lru_cache(None)
    def log_video(self, *args, **kwargs):
        self.log.warn("videos are not logged by std logger")
    
import os
class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir):
        # Check for existence of log_dir:
        # get the length of folders with same name:
        folder_name = log_dir
        i = 1
        while os.path.exists(folder_name):
            folder_name = f"{log_dir}_{i}"
            i += 1
        log_dir = folder_name
        self.writer = SummaryWriter(log_dir)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            self.writer.add_text(param, str(value), global_step=0)
    def log_history(self, param, value, step):
        self.writer.add_scalar(param, value, global_step=step)
    def log_video(self, video_path, name="video"):
        self.writer.add_video(name, video_path)
    def log_image(self, image_path, name="image"):
        self.writer.add_image(name, image_path)