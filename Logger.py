"""Logging utilities for experiment tracking."""
import json
import logging
import math
import os
import threading
import tempfile
from functools import lru_cache
from time import time
from typing import Optional

import wandb
from torch.utils.tensorboard import SummaryWriter

logger_types = {'wandb', 'std', 'tensorboard'}


class BaseLogger:
    """Base class for experiment loggers."""

    def __init__(self, run_dir: Optional[str] = None, history_flush_interval: float = 30.0):
        self.history = {}  # Store history: {metric: [(step, value, timestamp)]}
        self.start_time = time()
        self.run_dir = None
        self.history_path = None
        self.history_flush_interval = history_flush_interval
        self._last_flush_time = 0.0
        self._history_dirty = False
        self._history_lock = threading.Lock()
        if run_dir is not None:
            self.set_run_dir(run_dir)
            print("Logger enabled at", run_dir)

    def log_hparams(self, hparam_dict):
        """Log hyperparameters."""
        raise NotImplementedError()

    def log_history(self, param, value, step):
        """Log history metric."""
        raise NotImplementedError()

    def set_run_dir(self, run_dir: str) -> None:
        """Set the run directory and load existing history."""
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.history_path = os.path.join(run_dir, "history.json")
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for metric, entries in data.items():
                    self.history[metric] = [
                        (int(step), value, float(ts)) for step, value, ts in entries
                    ]
            except (json.JSONDecodeError, ValueError):
                pass

    def _store_history(self, param, value, step):
        """Store history locally for dashboard access."""
        # Sanitize value to prevent NaN/Infinity in JSON
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                value = None
        
        with self._history_lock:
            if param not in self.history:
                self.history[param] = []
            current_time = time() - self.start_time  # Time since training start
            self.history[param].append((step, value, current_time))
            self._history_dirty = True
        self._maybe_flush_history()

    def _maybe_flush_history(self):
        """Flush history to disk if interval elapsed."""
        if not self.history_path:
            return
        if not self._history_dirty:
            return
        now = time()
        if now - self._last_flush_time < self.history_flush_interval:
            return
        self._flush_history(now)

    def _flush_history(self, now: Optional[float] = None):
        """Persist cached history to disk."""
        if not self.history_path:
            return
        if now is None:
            now = time()

        # Take a snapshot of the history under lock and clear dirty flag.
        with self._history_lock:
            if not self._history_dirty:
                # update last flush time even if nothing to do
                self._last_flush_time = now
                return
            try:
                serializable = {
                    metric: [[step, value, ts] for step, value, ts in entries]
                    for metric, entries in self.history.items()
                }
            except Exception:
                serializable = {}
            # mark as flushed; new entries will set this True again
            self._history_dirty = False
            self._last_flush_time = now

        def _writer(snapshot, path):
            try:
                dirpath = os.path.dirname(path)
                os.makedirs(dirpath, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix="history_", suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f)
                    # atomic replace
                    os.replace(tmp_path, path)
                finally:
                    # ensure temp file removed if replace failed
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            except Exception:
                pass

        # Write in background to avoid blocking the main loop
        thread = threading.Thread(target=_writer, args=(serializable, self.history_path), daemon=True)
        thread.start()

    def log_video(self, video_path):
        """Log video."""
        raise NotImplementedError()

    def log_image(self, image_path):
        """Log image."""
        raise NotImplementedError()
    
    def close(self):
        """Close the logger."""
        self._flush_history()


class WandBLogger(BaseLogger):
    """Weights & Biases logger."""

    def __init__(self, entity, project, run_dir: Optional[str] = None):
        super().__init__(run_dir=run_dir)
        wandb.init(entity=entity, project=project)

    def log_hparams(self, hparam_dict):
        """Log hyperparameters."""
        for param, value in hparam_dict.items():
            try:
                wandb.log({param: value})
            except Exception:
                print(f"Could not log {param}: {value}")

    def log_history(self, param, value, step):
        """Log history metric."""
        self._store_history(param, value, step)
        wandb.log({param: value}, step=step)

    def log_video(self, video_path, name="video"):
        """Log video."""
        wandb.log({name: wandb.Video(video_path)})

    def log_image(self, image_path, name="image"):
        """Log image."""
        wandb.log({name: wandb.Image(image_path)})

    def close(self):
        """Close the WandB run."""
        super().close()
        try:
            wandb.finish()
        except Exception:
            pass


class StdLogger(BaseLogger):
    """Standard output logger."""

    def __init__(self, logger=None, run_dir: Optional[str] = None):
        super().__init__(run_dir=run_dir)
        if logger is not None:
            self.log = logger
        else:
            self.log = logging.getLogger("Barl")
            self.log.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)

    def log_hparams(self, hparam_dict):
        """Log hyperparameters."""
        for param, value in hparam_dict.items():
            self.log.info("%s: %s", param, value)

    def log_history(self, param, value, step):
        """Log history metric."""
        self._store_history(param, value, step)
        self.log.info("%s: %s", param, value)

    def log_video(self, video_path, name="video"):
        """Log video."""
        self.log.warning("videos are not logged by std logger")

    def log_image(self, image_path, name="image"):
        """Log image."""
        pass

    def close(self):
        """Close the logger."""
        super().close()


class TensorboardLogger(BaseLogger):
    """TensorBoard logger."""

    def __init__(self, log_dir, history_flush_interval: float = 10.0):
        folder_name = log_dir
        i = 1
        while os.path.exists(folder_name):
            folder_name = f"{log_dir}_{i}"
            i += 1
        self.writer = SummaryWriter(folder_name)
        super().__init__(run_dir=self.writer.log_dir, history_flush_interval=history_flush_interval)

    def log_hparams(self, hparam_dict):
        """Log hyperparameters."""
        for param, value in hparam_dict.items():
            self.writer.add_text(param, str(value), global_step=0)
        with open(os.path.join(self.writer.log_dir, "hparams.txt"), "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {time()}\nHyperparameters:\n")
            for param, value in hparam_dict.items():
                f.write(f"{param}: {value}\n")

    def log_history(self, param, value, step):
        """Log history metric."""
        self._store_history(param, value, step)
        self.writer.add_scalar(param, value, global_step=step)

    def log_video(self, video_path, name="video"):
        """Log video."""
        self.writer.add_video(name, video_path)

    def log_image(self, image_path, name="image"):
        """Log image."""
        self.writer.add_image(name, image_path)

    def close(self):
        """Close the TensorBoard writer."""
        super().close()
        self.writer.close()