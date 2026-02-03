"""Video recording utilities for evaluation episodes."""
import json
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from utils import make_video_env


class VideoRecorder:
    """Handles video recording of evaluation episodes."""

    def __init__(
        self,
        record_enabled: bool = False,
        record_every: int = 1,
        num_episodes: int = 1,
        async_recording: bool = False,
    ):
        self.record_enabled = record_enabled
        self.record_every = max(1, int(record_every))
        self.num_episodes = max(1, int(num_episodes))
        self.async_recording = async_recording

        self._eval_count = 0
        self._video_index = 0
        self._recording = False
        self._lock = threading.Lock()
        self._active_threads = []

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

    def record_async(
        self,
        env_id,
        video_dir: str,
        step: int,
        evaluation_policy: Callable,
        is_atari: bool = False,
        permute_dims: bool = False,
        log_callback: Optional[Callable[[str, float, int], None]] = None,
    ) -> None:
        """Start async video recording in background thread."""
        with self._lock:
            if self._recording:
                return
            self._recording = True

        def worker():
            try:
                self._record_episodes(
                    env_id, video_dir, step, evaluation_policy, is_atari, permute_dims, log_callback
                )
            finally:
                with self._lock:
                    self._recording = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with self._lock:
            self._active_threads.append(thread)

    def _record_episodes(
        self,
        env_id,
        video_dir: str,
        step: int,
        evaluation_policy: Callable,
        is_atari: bool,
        permute_dims: bool,
        log_callback: Optional[Callable[[str, float, int], None]] = None,
    ) -> None:
        """Record evaluation episodes (internal worker)."""
        video_env, name_prefix = self.get_video_env(
            env_id, video_dir, step, is_atari, permute_dims
        )

        episode_rewards = []
        try:
            for _ in range(self.num_episodes):
                state, _ = video_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action = evaluation_policy(state)
                    next_state, reward, terminated, truncated, _ = video_env.step(action)
                    ep_reward += reward
                    state = next_state
                    done = terminated or truncated
                episode_rewards.append(ep_reward)
        finally:
            try:
                video_env.close()
            except Exception:
                pass

        if episode_rewards:
            avg_reward = float(np.mean(episode_rewards))
            self._save_video_metadata(video_dir, name_prefix, avg_reward, step)
            if log_callback:
                log_callback('eval/video_avg_reward', avg_reward, step)

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
        with self._lock:
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

    def wait_for_completion(self, timeout: Optional[float] = None) -> None:
        """Wait for all active recording threads to complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
        """
        with self._lock:
            threads = self._active_threads.copy()
        
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=timeout)
        
        with self._lock:
            # Clean up completed threads
            self._active_threads = [t for t in self._active_threads if t.is_alive()]


def resolve_video_dir(loggers) -> Optional[str]:
    """Find the first logger with a run_dir and create videos subdirectory."""
    for logger in loggers:
        run_dir = getattr(logger, "run_dir", None)
        if run_dir:
            video_dir = os.path.join(run_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            return video_dir
    return None
