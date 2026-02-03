"""Video recording utilities for evaluation episodes."""
import json
import os
import time
from typing import Optional

from utils import make_video_env


class VideoRecorder:
    """Handles video recording of evaluation episodes."""

    def __init__(
        self,
        record_enabled: bool = False,
        record_every: int = 1,
        num_episodes: int = 1,
    ):
        self.record_enabled = record_enabled
        self.record_every = max(1, int(record_every))
        self.num_episodes = max(1, int(num_episodes))

        self._eval_count = 0
        self._video_index = 0

    def should_record(self) -> bool:
        """Check if this evaluation should be recorded."""
        self._eval_count += 1
        return self.record_enabled and (self._eval_count % self.record_every == 0)

    def get_video_env(
        self,
        env_id,
        video_dir: str,
        step: int,
        is_atari: bool = False,
        permute_dims: bool = False,
    ):
        """Create a video-capable evaluation environment."""
        name_prefix = f"eval_step_{step}_vid_{self._video_index}"
        self._video_index += 1

        def episode_trigger(episode_id: int) -> bool:
            return episode_id < self.num_episodes

        return (
            make_video_env(
                env_id,
                video_folder=video_dir,
                name_prefix=name_prefix,
                is_atari=is_atari,
                permute_dims=permute_dims,
                episode_trigger=episode_trigger,
            ),
            name_prefix,
        )

    def _save_video_metadata(
        self, video_dir: str, name_prefix: str, avg_reward: float, step: int
    ) -> None:
        """Persist video metadata for dashboard display."""
        metadata_path = os.path.join(video_dir, "video_metadata.json")
        entry = {
            "prefix": name_prefix,
            "avg_reward": avg_reward,
            "step": step,
            "timestamp": time.time(),
        }
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
        except Exception:
            data = []
        data.append(entry)
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass


def resolve_video_dir(loggers) -> Optional[str]:
    """Find the first logger with a run_dir and create videos subdirectory."""
    for logger in loggers:
        run_dir = getattr(logger, "run_dir", None)
        if run_dir:
            video_dir = os.path.join(run_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            return video_dir
    return None
