import json
import os
import queue
import socket
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS


class DashboardGUI:
    """
    Modular web dashboard for visualizing and controlling RL agent training.
    Acts as a hub for multiple runs with notes and file access.
    """

    def __init__(self, agent=None, base_log_dir: str = "logs"):
        self.agent = agent
        self.app = None
        self.server_thread = None
        self.is_running = False
        self.port = 8050

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dashboard_dir = os.path.join(self.base_dir, "dashboard")
        self.template_dir = os.path.join(self.dashboard_dir, "templates")
        self.static_dir = os.path.join(self.dashboard_dir, "static")
        self.base_log_dir = os.path.abspath(os.path.join(self.base_dir, base_log_dir))
        self.uploaded_log_dir = os.path.abspath(os.path.join(self.base_dir, "uploaded_logs"))

        self.metrics_data = defaultdict(lambda: {"steps": [], "values": [], "times": []})
        self.available_metrics = set()
        self.x_axis_mode = "time"  # "time", "steps", or "episodes"

        self.command_queue = queue.Queue()
        self.agent_state = {
            "paused": False,
            "learning_rate": getattr(agent, "learning_rate", None) if agent else None,
            "epsilon": getattr(agent, "epsilon", None) if agent else None,
            "total_steps": 0,
        }
        
        # Cache for runs list to avoid rescanning directories too frequently
        self._runs_cache = None
        self._runs_cache_time = 0
        self._runs_cache_ttl = 2.0  # Cache for 2 seconds

        self.current_run_dir = self._resolve_current_run_dir()
        self.current_run_id = self._resolve_current_run_id()

        self._initialize_app()

    def _initialize_app(self):
        self.app = Flask(
            __name__,
            template_folder=self.template_dir,
            static_folder=self.static_dir,
        )
        CORS(self.app)
        self._setup_routes()

        # Ensure uploaded logs directory exists
        os.makedirs(self.uploaded_log_dir, exist_ok=True)

    def _resolve_current_run_dir(self) -> Optional[str]:
        if not self.agent:
            return None
        for logger in getattr(self.agent, "loggers", []):
            run_dir = getattr(logger, "run_dir", None)
            if run_dir:
                return os.path.abspath(run_dir)
        return None

    def _resolve_current_run_id(self) -> Optional[str]:
        if not self.current_run_dir:
            # Try to find most recent run (history.json modified within last 10 seconds)
            # Force bypass cache to get fresh filesystem data for current run detection
            runs = self._list_runs(use_cache=False)
            current_time = time.time()
            for run in runs:
                # Check history.json modification time instead of directory
                history_path = os.path.join(run["path"], "history.json")
                if os.path.exists(history_path):
                    history_mtime = os.path.getmtime(history_path)
                    if current_time - history_mtime < 10:
                        return run["id"]
            return None
        if self.current_run_dir.startswith(self.base_log_dir + os.sep):
            return os.path.relpath(self.current_run_dir, self.base_log_dir)
        return os.path.basename(self.current_run_dir)

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template(
                "index.html",
                runs=self._list_runs(),
                current_run_id=self.current_run_id,
            )

        @self.app.route("/run/<path:run_id>")
        def run_view(run_id: str):
            # Read algo_name and env_str from the run directory
            run_dir = self._run_dir_from_id(run_id)
            algo_name = "N/A"
            env_name = "N/A"
            
            if run_dir:
                algo_name_path = os.path.join(run_dir, "algo_name.txt")
                if os.path.exists(algo_name_path):
                    try:
                        with open(algo_name_path, "r") as f:
                            algo_name = f.read().strip()
                    except Exception:
                        pass
                
                env_str_path = os.path.join(run_dir, "env_str.txt")
                if os.path.exists(env_str_path):
                    try:
                        with open(env_str_path, "r") as f:
                            env_name = f.read().strip()
                    except Exception:
                        pass
            
            return render_template(
                "run.html",
                run_id=run_id,
                current_run_id=self.current_run_id,
                env_name=env_name,
                agent_class=algo_name,
            )

        @self.app.route("/api/runs")
        def api_runs():
            # Update current run ID dynamically
            self.current_run_id = self._resolve_current_run_id()
            return jsonify(
                {
                    "runs": self._list_runs(),
                    "current_run_id": self.current_run_id,
                }
            )

        @self.app.route("/api/upload_logs", methods=["POST"])
        def api_upload_logs():
            files = request.files.getlist("files")
            if not files:
                return jsonify({"error": "No files uploaded"}), 400

            run_ids = set()
            os.makedirs(self.uploaded_log_dir, exist_ok=True)

            for storage_file in files:
                raw_name = (storage_file.filename or "").replace("\\", "/")
                if not raw_name:
                    continue
                normalized = os.path.normpath(raw_name)
                if normalized.startswith("..") or os.path.isabs(normalized):
                    continue

                segments = normalized.split(os.sep)
                run_name = segments[0] if segments else "uploaded_run"
                if not run_name:
                    run_name = "uploaded_run"

                rel_inside = "/".join(segments[1:]) if len(segments) > 1 else os.path.basename(normalized)
                if not rel_inside:
                    rel_inside = os.path.basename(normalized)

                run_dir = os.path.join(self.uploaded_log_dir, run_name)
                dest_path = os.path.abspath(os.path.join(run_dir, rel_inside))
                if not dest_path.startswith(run_dir + os.sep) and dest_path != run_dir:
                    continue

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                storage_file.save(dest_path)
                run_ids.add(f"uploaded/{run_name}")

            print(f"[upload] received {len(files)} files, runs={sorted(run_ids)}")

            # Build history.json from TensorBoard event files for each run
            for run_id in sorted(run_ids):
                run_dir = self._run_dir_from_id(run_id)
                if not run_dir:
                    print(f"[upload] run dir not found for {run_id}")
                    continue
                try:
                    history = self._build_history_from_tfevents(run_dir)
                    if history:
                        history_path = os.path.join(run_dir, "history.json")
                        with open(history_path, "w") as f:
                            json.dump(history, f)
                        print(f"[upload] wrote history.json for {run_id} with {len(history)} metrics")
                    else:
                        print(f"[upload] no scalar data found for {run_id}")
                except Exception as exc:
                    print(f"[upload] failed to build history for {run_id}: {exc}")

            # Bust cache so newly uploaded runs appear immediately
            self._runs_cache = None
            self._runs_cache_time = 0

            return jsonify({"success": True, "run_ids": sorted(run_ids)})

        @self.app.route("/api/runs/<path:run_id>/metrics")
        def api_run_metrics(run_id: str):
            # Always load from history.json file
            data = self._load_history_from_run(run_id)
            
            # If this is the current run with an active agent, also collect from loggers
            if self._is_current_run(run_id):
                self._collect_metrics_from_loggers()
                # Merge logger data with history data
                for metric, vals in self.metrics_data.items():
                    data[metric] = {
                        "steps": list(vals["steps"]),
                        "values": list(vals["values"]),
                        "times": list(vals["times"]),
                    }

            return jsonify(
                {
                    "metrics": sorted(list(data.keys())),
                    "x_axis_mode": self.x_axis_mode,
                    "data": data,
                }
            )

        @self.app.route("/api/runs/<path:run_id>/notes", methods=["GET", "POST"])
        def api_run_notes(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            notes_path = os.path.join(run_dir, "notes.md")

            if request.method == "POST":
                data = request.get_json() or {}
                notes = data.get("notes", "")
                with open(notes_path, "w") as f:
                    f.write(notes)
                return jsonify({"success": True})

            if os.path.exists(notes_path):
                with open(notes_path, "r") as f:
                    return jsonify({"notes": f.read()})
            return jsonify({"notes": ""})

        @self.app.route("/api/runs/<path:run_id>/buffer_stats")
        def api_run_buffer_stats(run_id: str):
            # Load buffer statistics from history
            data = self._load_history_from_run(run_id)
            buffer_stats = {
                "n_stored": data.get("buffer/n_stored", {}),
                "terminated_fraction": data.get("buffer/terminated_fraction", {}),
            }
            # Load reward histogram
            run_dir = self._run_dir_from_id(run_id)
            if run_dir:
                histogram_path = os.path.join(run_dir, "reward_histogram.json")
                if os.path.exists(histogram_path):
                    try:
                        with open(histogram_path, "r") as f:
                            buffer_stats["reward_histogram"] = json.load(f)
                    except Exception:
                        pass
            return jsonify(buffer_stats)

        @self.app.route("/api/runs/<path:run_id>/hparams")
        def api_run_hparams(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            hparams_path = os.path.join(run_dir, "hparams.txt")
            if not os.path.exists(hparams_path):
                return jsonify({"hparams": {}})
            try:
                hparams = {}
                with open(hparams_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[2:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        hparams[key.strip()] = value.strip()
                return jsonify({"hparams": hparams})
            except Exception:
                return jsonify({"hparams": {}})

        @self.app.route("/api/runs/<path:run_id>/storage")
        def api_run_storage(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            breakdown = self._calculate_run_storage(run_dir)
            return jsonify(breakdown)

        @self.app.route("/api/runs/<path:run_id>/videos")
        def api_run_videos(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"videos": []})
            video_dir = os.path.join(run_dir, "videos")
            if not os.path.isdir(video_dir):
                return jsonify({"videos": []})
            metadata = []
            metadata_path = os.path.join(video_dir, "video_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = []
            prefix_to_entry = {}
            for entry in metadata:
                prefix = entry.get("prefix")
                if not prefix:
                    continue
                prefix_to_entry[prefix] = entry
            videos = []
            for name in sorted(os.listdir(video_dir)):
                if not name.lower().endswith((".mp4", ".webm", ".gif")):
                    continue
                path = os.path.join(video_dir, name)
                if not os.path.isfile(path):
                    continue
                reward = None
                for prefix, entry in prefix_to_entry.items():
                    if prefix in name:
                        reward = entry.get("avg_reward")
                        break
                videos.append(
                    {
                        "name": name,
                        "size": os.path.getsize(path),
                        "mtime": os.path.getmtime(path),
                        "url": f"/api/runs/{run_id}/videos/{name}",
                        "reward": reward,
                    }
                )
            videos.sort(key=lambda v: v["mtime"], reverse=True)
            return jsonify({"videos": videos})

        @self.app.route("/api/runs/<path:run_id>/videos/<path:filename>")
        def api_run_video_file(run_id: str, filename: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            video_dir = os.path.join(run_dir, "videos")
            if not os.path.isdir(video_dir):
                return jsonify({"error": "Video directory not found"}), 404
            safe_path = os.path.abspath(os.path.join(video_dir, filename))
            if not safe_path.startswith(video_dir + os.sep):
                return jsonify({"error": "Invalid path"}), 400
            if not os.path.isfile(safe_path):
                return jsonify({"error": "Video not found"}), 404
            return send_from_directory(video_dir, filename, as_attachment=False)
            try:
                hparams = {}
                with open(hparams_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[2:]:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            hparams[key.strip()] = value.strip()
                return jsonify({"hparams": hparams})
            except Exception:
                return jsonify({"hparams": {}})

        @self.app.route("/api/runs/<path:run_id>/files")
        def api_run_files(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            files = []
            for name in sorted(os.listdir(run_dir)):
                path = os.path.join(run_dir, name)
                if os.path.isfile(path):
                    files.append({
                        "name": name,
                        "size": os.path.getsize(path),
                    })
            return jsonify({"files": files})

        @self.app.route("/api/agent/state")
        def get_agent_state():
            if not self.agent:
                return jsonify({})
            self.agent_state.update(
                {
                    "learning_rate": getattr(self.agent, "learning_rate", None),
                    "epsilon": getattr(self.agent, "epsilon", None),
                    "total_steps": getattr(self.agent, "total_env_steps", 0),
                    "learn_steps": getattr(self.agent, "learn_env_steps", 0),
                    "num_episodes": getattr(self.agent, "num_episodes", 0),
                }
            )
            return jsonify(self.agent_state)

        @self.app.route("/api/agent/command", methods=["POST"])
        def send_command():
            if not self.agent:
                return jsonify({"error": "No agent attached"}), 400
            data = request.get_json() or {}
            command = data.get("command")
            params = data.get("params", {})

            if not command:
                return jsonify({"error": "No command provided"}), 400

            self.command_queue.put({"command": command, "params": params})
            result = self._execute_command(command, params)
            return jsonify({"success": True, "result": result})

        @self.app.route("/api/axis/toggle", methods=["POST"])
        def toggle_axis():
            if self.x_axis_mode == "time":
                self.x_axis_mode = "steps"
            elif self.x_axis_mode == "steps":
                self.x_axis_mode = "episodes"
            else:
                self.x_axis_mode = "time"
            return jsonify({"x_axis_mode": self.x_axis_mode})

        @self.app.route("/api/stream")
        def stream():
            if not self.agent:
                return Response("", mimetype="text/event-stream")
            def event_stream():
                last_step = 0
                while self.is_running:
                    self._collect_metrics_from_loggers()
                    current_step = getattr(self.agent, "learn_env_steps", 0)

                    if current_step > last_step:
                        data = {
                            "step": current_step,
                            "metrics": {
                                metric: {
                                    "latest": vals["values"][-1] if vals["values"] else None
                                }
                                for metric, vals in self.metrics_data.items()
                            },
                            "agent_state": {
                                "epsilon": getattr(self.agent, "epsilon", None),
                                "learning_rate": getattr(self.agent, "learning_rate", None),
                            },
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        last_step = current_step

                    time.sleep(1)

            return Response(event_stream(), mimetype="text/event-stream")

    def _is_current_run(self, run_id: str) -> bool:
        # First check if this is the agent's own run
        if self.current_run_id is not None and run_id == self.current_run_id:
            return True
        # Also check if this run has been updated recently (within 10 seconds)
        run_dir = self._run_dir_from_id(run_id)
        if run_dir:
            history_path = os.path.join(run_dir, "history.json")
            if os.path.exists(history_path):
                history_mtime = os.path.getmtime(history_path)
                if time.time() - history_mtime < 10:
                    return True
        return False

    def _run_dir_from_id(self, run_id: str) -> Optional[str]:
        if run_id.startswith("uploaded/"):
            uploaded_id = run_id.split("/", 1)[1]
            run_dir = os.path.abspath(os.path.join(self.uploaded_log_dir, uploaded_id))
            if os.path.isdir(run_dir) and run_dir.startswith(self.uploaded_log_dir):
                return run_dir

        run_dir = os.path.abspath(os.path.join(self.base_log_dir, run_id))
        if os.path.isdir(run_dir) and run_dir.startswith(self.base_log_dir):
            return run_dir
        return None

    def _list_runs(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        # Check cache first
        if use_cache and self._runs_cache is not None:
            if time.time() - self._runs_cache_time < self._runs_cache_ttl:
                return self._runs_cache
        
        runs = []

        def add_runs_from_root(root_dir: str, prefix: Optional[str] = None):
            if not os.path.isdir(root_dir):
                return
            for name in sorted(os.listdir(root_dir)):
                path = os.path.join(root_dir, name)
                if not os.path.isdir(path):
                    continue
                algo_name_path = os.path.join(path, "algo_name.txt")
                algo_name = "unknown"
                if os.path.exists(algo_name_path):
                    try:
                        with open(algo_name_path, "r") as f:
                            algo_name = f.read().strip()
                    except Exception:
                        pass
                env_str_path = os.path.join(path, "env_str.txt")
                env_str = "unknown"
                if os.path.exists(env_str_path):
                    try:
                        with open(env_str_path, "r") as f:
                            env_str = f.read().strip()
                    except Exception:
                        pass
                run_id = f"{prefix}/{name}" if prefix else name
                runs.append(
                    {
                        "id": run_id,
                        "path": path,
                        "algo_name": algo_name,
                        "env_str": env_str,
                        "mtime": os.path.getmtime(path),
                    }
                )

        add_runs_from_root(self.base_log_dir)
        add_runs_from_root(self.uploaded_log_dir, prefix="uploaded")
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        
        # Update cache
        self._runs_cache = runs
        self._runs_cache_time = time.time()
        return runs

    def _load_history_from_run(self, run_id: str) -> Dict[str, Dict[str, List[Any]]]:
        run_dir = self._run_dir_from_id(run_id)
        if not run_dir:
            return {}
        
        # For uploaded runs, re-parse event files every time to catch live updates
        if run_id.startswith("uploaded/"):
            print(f"[metrics] re-parsing event files for {run_id}")
            history = self._build_history_from_tfevents(run_dir)
            if history:
                history_path = os.path.join(run_dir, "history.json")
                try:
                    with open(history_path, "w") as f:
                        json.dump(history, f)
                except Exception as exc:
                    print(f"[metrics] failed to write history.json: {exc}")
                return {
                    metric: {
                        "steps": [entry[0] for entry in entries],
                        "values": [entry[1] for entry in entries],
                        "times": [entry[2] for entry in entries],
                    }
                    for metric, entries in history.items()
                }
        
        # For non-uploaded runs, use cached history.json
        history_path = os.path.join(run_dir, "history.json")
        if not os.path.exists(history_path):
            return {}
        try:
            with open(history_path, "r") as f:
                data = json.load(f)
            return {
                metric: {
                    "steps": [entry[0] for entry in entries],
                    "values": [entry[1] for entry in entries],
                    "times": [entry[2] for entry in entries],
                }
                for metric, entries in data.items()
            }
        except Exception:
            return {}

    def _build_history_from_tfevents(self, run_dir: str) -> Dict[str, List[List[Any]]]:
        event_files = []
        for root, _dirs, files in os.walk(run_dir):
            for name in files:
                if "tfevents" in name or name.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, name))

        if not event_files:
            return {}

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception as exc:
            print(f"[upload] tensorboard import failed: {exc}")
            return {}

        merged: Dict[str, Dict[str, List[Any]]] = {}

        for event_file in event_files:
            try:
                acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
                acc.Reload()
                tags = acc.Tags().get("scalars", [])
                for tag in tags:
                    scalars = acc.Scalars(tag)
                    if tag not in merged:
                        merged[tag] = {"steps": [], "values": [], "times": []}
                    for s in scalars:
                        merged[tag]["steps"].append(s.step)
                        merged[tag]["values"].append(float(s.value))
                        merged[tag]["times"].append(float(s.wall_time))
            except Exception as exc:
                print(f"[upload] failed to read {event_file}: {exc}")

        # Convert to history.json format: metric -> list of [step, value, time]
        history: Dict[str, List[List[Any]]] = {}
        for tag, vals in merged.items():
            triples = list(zip(vals["steps"], vals["values"], vals["times"]))
            triples.sort(key=lambda t: t[0])
            history[tag] = [[int(step), float(value), float(ts)] for step, value, ts in triples]

        return history

    def _calculate_run_storage(self, run_dir: str) -> Dict[str, Any]:
        totals = {
            "videos": 0,
            "metrics": 0,
            "models": 0,
            "images": 0,
            "notes": 0,
            "other": 0,
        }
        total = 0

        video_dir = os.path.join(run_dir, "videos")
        for root, _dirs, files in os.walk(run_dir):
            for name in files:
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                except Exception:
                    continue
                total += size

                rel = os.path.relpath(path, run_dir)
                lower = name.lower()

                if root.startswith(video_dir):
                    totals["videos"] += size
                elif lower.endswith((".pt", ".pth", ".ckpt")):
                    totals["models"] += size
                elif lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    totals["images"] += size
                elif lower.endswith((".json", ".tfevents", ".event", ".events")) or "history.json" in lower:
                    totals["metrics"] += size
                elif rel == "notes.md":
                    totals["notes"] += size
                else:
                    totals["other"] += size

        return {
            "total": total,
            "breakdown": totals,
        }

    def _execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if command == "evaluate":
                n_episodes = int(params.get("n_episodes", 10))
                avg_reward = self.agent.evaluate(n_episodes=n_episodes)
                return {"message": f"Evaluation complete: {avg_reward:.2f}", "reward": avg_reward}

            if command == "save":
                path = params.get("path", f"./checkpoints/{self.agent.__class__.__name__}_checkpoint.pt")
                self.agent.save(path)
                return {"message": f"Model saved to {path}"}

            return {"error": f"Unknown command: {command}"}
        except Exception as e:
            return {"error": str(e)}

    def _collect_metrics_from_loggers(self):
        if not self.agent:
            return
        for logger in getattr(self.agent, "loggers", []):
            if hasattr(logger, "history") and logger.history:
                for metric_name, values in logger.history.items():
                    self.available_metrics.add(metric_name)
                    if metric_name not in self.metrics_data:
                        self.metrics_data[metric_name] = {"steps": [], "values": [], "times": []}
                    current_steps = set(self.metrics_data[metric_name]["steps"])
                    for step, value, timestamp in values:
                        if step not in current_steps:
                            self.metrics_data[metric_name]["steps"].append(step)
                            self.metrics_data[metric_name]["values"].append(value)
                            self.metrics_data[metric_name]["times"].append(timestamp)
                            current_steps.add(step)

    def launch_dashboard_gui(self, port: int = 8050):
        if self.is_running:
            print(f"Dashboard is already running at http://localhost:{port}")
            return
        if self._is_port_in_use(port):
            print(f"Dashboard already running at http://localhost:{port}")
            return

        self.port = port

        def run_server():
            self.is_running = True
            print(f"\n{'=' * 70}")
            print(f"🚀 Dashboard Hub: http://localhost:{port}")
            if self.current_run_id:
                print(f"   Current run: {self.current_run_id}")
            print(f"{'=' * 70}\n")
            self.app.run(debug=False, port=port, use_reloader=False, threaded=True)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)

    def update_dashboard_gui(self, data: dict) -> None:
        pass

    def stop(self):
        self.is_running = False
        if self.server_thread:
            print("\n🛑 Dashboard stopped.")


if __name__ == "__main__":
    print("This module requires an agent instance to run.")
    print("Example usage:")
    print("  from DQN import DQN")
    print("  agent = DQN('CartPole-v1')")
    print("  agent.launch_dashboard_gui(port=8050)")
    print("  agent.learn(total_timesteps=10000)")
    