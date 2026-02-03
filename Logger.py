# Wandb, tensorboard, stdout, python logger
from functools import lru_cache
from time import time
from torch.utils.tensorboard import SummaryWriter
import logging
import wandb
import os
import json
from typing import Optional



logger_types = {'wandb', 'std', 'tensorboard'}


class BaseLogger:
    def __init__(self, run_dir: Optional[str] = None, history_flush_interval: float = 2.0):
        self.history = {}  # Store history: {metric: [(step, value, timestamp)]}
        self.start_time = time()
        self.run_dir = None
        self.history_path = None
        self.history_flush_interval = history_flush_interval
        self._last_flush_time = 0.0
        if run_dir is not None:
            self.set_run_dir(run_dir)
    
    def log_hparams(self, hparam_dict):
        raise NotImplementedError()
    
    def log_history(self, param, value, step):
        raise NotImplementedError()
    
    def set_run_dir(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.history_path = os.path.join(run_dir, "history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                for metric, entries in data.items():
                    self.history[metric] = [
                        (int(step), value, float(ts)) for step, value, ts in entries
                    ]
            except Exception:
                pass

    def _store_history(self, param, value, step):
        """Store history locally for dashboard access"""
        if param not in self.history:
            self.history[param] = []
        current_time = time() - self.start_time  # Time since training start
        self.history[param].append((step, value, current_time))
        self._maybe_flush_history()

    def _maybe_flush_history(self):
        if not self.history_path:
            return
        now = time()
        if now - self._last_flush_time < self.history_flush_interval:
            return
        self._last_flush_time = now
        try:
            serializable = {
                metric: [[step, value, ts] for step, value, ts in entries]
                for metric, entries in self.history.items()
            }
            with open(self.history_path, "w") as f:
                json.dump(serializable, f)
        except Exception:
            pass
    
    def log_video(self, video_path):
        raise NotImplementedError()
    
    def log_image(self, image_path):
        raise NotImplementedError()
        

class WandBLogger(BaseLogger):
    def __init__(self, entity, project, run_dir: Optional[str] = None):
        super().__init__(run_dir=run_dir)
        wandb.init(entity=entity, project=project)
    
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            # check if not serializable:
            try:
                wandb.log({param: value})
            except Exception as e:
                print(f"Could not log {param}: {value}")
    def log_history(self, param, value, step):
        self._store_history(param, value, step)
        wandb.log({param: value}, step=step)
    def log_video(self, video_path, name="video"):
        wandb.log({name: wandb.Video(video_path)})
    def log_image(self, image_path, name="image"):
        wandb.log({name: wandb.Image(image_path)})


class StdLogger(BaseLogger):
    def __init__(self, logger=None, run_dir: Optional[str] = None):
        super().__init__(run_dir=run_dir)
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
        self._store_history(param, value, step)
        self.log.info(f"{param}: {value}")
    @lru_cache(None)
    def log_video(self, *args, **kwargs):
        self.log.warning("videos are not logged by std logger")
    
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
        super().__init__(run_dir=self.writer.log_dir)
    def log_hparams(self, hparam_dict):
        for param, value in hparam_dict.items():
            self.writer.add_text(param, str(value), global_step=0)
        # Also store them in the same folder that the logger uses:
        with open(os.path.join(self.writer.log_dir, "hparams.txt"), "w") as f:
            # On the first line, write the timestamp and name of the logger:
            f.write(f"Timestamp: {time()}\nHyperparameters:\n")
            for param, value in hparam_dict.items():
                f.write(f"{param}: {value}\n")
        # Save algo_name and env_str separately for dashboard
        algo_name = hparam_dict.get("algo_name", "unknown")
        with open(os.path.join(self.writer.log_dir, "algo_name.txt"), "w") as f:
            f.write(algo_name)
        env_str = hparam_dict.get("env_str", "unknown")
        with open(os.path.join(self.writer.log_dir, "env_str.txt"), "w") as f:
            f.write(env_str)

    def log_history(self, param, value, step):
        self._store_history(param, value, step)
        self.writer.add_scalar(param, value, global_step=step)
    def log_video(self, video_path, name="video"):
        self.writer.add_video(name, video_path)
    def log_image(self, image_path, name="image"):
        self.writer.add_image(name, image_path)
    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, 'writer'):
            self.writer.close()