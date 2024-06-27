# Wandb, tensorboard, stdout, python logger
from functools import lru_cache

import logging
import wandb


logger_types = {'wandb', 'std', 'tensorboard'}


class BaseLogger: 
    def log_hparams(self, hparam_dict):
        raise NotImplementedError()
    def log_history(self, param, value):
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
            wandb.log({param: value})
    def log_history(self, param, value):
        wandb.log({param: value})
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
    def log_history(self, param, value):
        self.log.info(f"{param}: {value}")
    @lru_cache(None)
    def log_video(self, *args, **kwargs):
        self.log.warn("videos are not logged by std logger")
    