import json
import math
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS


class DashboardGUI:
    """
    Web dashboard for visualizing and monitoring RL agent training runs.
    Runs independently and allows browsing multiple training runs via directory selection.
    """

    def __init__(self, base_log_dir: str = "logs"):
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

        self.x_axis_mode = "steps"  # "time", "steps", or "episodes"

        # Cache for runs list to avoid rescanning directories too frequently
        self._runs_cache = None
        self._runs_cache_time = 0
        self._runs_cache_ttl = 2.0  # Cache for 2 seconds

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



    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        """Convert NaN and Infinity values to None for JSON serialization."""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template(
                "index.html",
                runs=self._list_runs(),
            )

        @self.app.route("/run/<path:run_id>")
        def run_view(run_id: str):
            # Read algo_name and env_str from run_data.json
            run_data = self._load_run_data(run_id)
            algo_name = run_data.get("algo_name", "N/A")
            env_name = run_data.get("env_str", "N/A")
            
            return render_template(
                "run.html",
                run_id=run_id,
                env_name=env_name,
                agent_class=algo_name,
            )

        @self.app.route("/api/runs")
        def api_runs():
            return jsonify(
                {
                    "runs": self._list_runs(),
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
            # Load from history.json file
            data = self._load_history_from_run(run_id)
            
            # Load run_data.json to get named_networks for categorization
            run_data = self._load_run_data(run_id)
            named_networks = run_data.get('named_networks', [])
            
            # Categorize metrics
            categorized = self._categorize_metrics(list(data.keys()), named_networks)

            return jsonify(
                {
                    "metrics": sorted(list(data.keys())),
                    "categorized_metrics": categorized,
                    "x_axis_mode": self.x_axis_mode,
                    "data": data,
                }
            )

        @self.app.route("/api/runs/<path:run_id>/status")
        def api_run_status(run_id: str):
            run_dir = self._run_dir_from_id(run_id)
            if not run_dir:
                return jsonify({"error": "Run not found"}), 404
            status = self._get_run_status(run_dir)
            return jsonify({"status": status})

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

        @self.app.route("/api/axis/toggle", methods=["POST"])
        def toggle_axis():
            if self.x_axis_mode == "time":
                self.x_axis_mode = "steps"
            elif self.x_axis_mode == "steps":
                self.x_axis_mode = "episodes"
            else:
                self.x_axis_mode = "time"
            return jsonify({"x_axis_mode": self.x_axis_mode})



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

    def _get_run_status(self, run_dir: str) -> str:
        """Get the status of a run from run_data.json."""
        run_data_path = os.path.join(run_dir, "run_data.json")
        if os.path.exists(run_data_path):
            try:
                with open(run_data_path, "r") as f:
                    data = json.load(f)
                status = data.get("status", "stopped")
                return status
            except Exception:
                pass
        return "stopped"

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
                run_id = f"{prefix}/{name}" if prefix else name
                run_data = self._load_run_data(run_id)
                algo_name = run_data.get("algo_name", "unknown")
                env_str = run_data.get("env_str", "N/A")
                status = run_data.get("status", "stopped")
                if status not in ["running", "stopped"]:
                    status = "stopped"
                runs.append(
                    {
                        "id": run_id,
                        "path": path,
                        "algo_name": algo_name,
                        "env_str": env_str,
                        "mtime": os.path.getmtime(path),
                        "status": status,
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
                        "values": [self._sanitize_value(entry[1]) for entry in entries],
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
                    "values": [self._sanitize_value(entry[1]) for entry in entries],
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
                elif lower.endswith((".json", ".event", ".events")) or "history.json" in lower or "tfevents" in lower:
                    totals["metrics"] += size
                elif rel == "notes.md":
                    totals["notes"] += size
                else:
                    totals["other"] += size

        return {
            "total": total,
            "breakdown": totals,
        }

    def _load_run_data(self, run_id: str) -> Dict[str, Any]:
        """Load run_data.json which contains named_networks and other metadata."""
        run_dir = self._run_dir_from_id(run_id)
        if not run_dir:
            return {}
        
        run_data_path = os.path.join(run_dir, "run_data.json")
        if not os.path.exists(run_data_path):
            return {}
        
        try:
            with open(run_data_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _categorize_metrics(self, metrics: List[str], named_networks: List[str]) -> Dict[str, List[str]]:
        """
        Categorize metrics into:
        - Training stats (C): anything not in network or buffer stats
        - Buffer stats (B): anything with *buffer*/*
        - Network stats (A): anything starting with a named network prefix
        
        Returns dict with keys 'training', 'buffer', 'network' in that order.
        """
        training = []
        buffer = []
        network = []
        
        for metric in metrics:
            # Check if it's a network metric (starts with a named network prefix)
            is_network = False
            for net_name in named_networks:
                if metric.startswith(f"{net_name}/"):
                    network.append(metric)
                    is_network = True
                    break
            
            if is_network:
                continue
            
            # Check if it's a buffer metric (contains 'buffer')
            if 'buffer' in metric:
                buffer.append(metric)
                continue
            
            # Everything else is training
            training.append(metric)
        
        return {
            'training': sorted(training),
            'buffer': sorted(buffer),
            'network': sorted(network),
        }

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
            print(f"{'=' * 70}\n")
            self.app.run(debug=False, port=port, use_reloader=False, threaded=True)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)

    def stop(self):
        self.is_running = False
        if self.server_thread:
            print("\n🛑 Dashboard stopped.")


if __name__ == "__main__":
    print("Dashboard GUI")
    print("Example usage:")
    print("  dashboard = DashboardGUI(base_log_dir='logs')")
    print("  dashboard.launch_dashboard_gui(port=8050)")
    