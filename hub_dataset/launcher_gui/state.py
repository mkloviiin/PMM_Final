"""Estado global compartido del Hub.

Estas variables son deliberadamente mutables a nivel de modulo: session.py,
curation.py, hub_push.py y los backends de robot en robots/ las leen y
escriben directamente como atributos de este modulo (`state.running = True`),
sin declaraciones `global` -- alcanza con mutar el modulo importado.
"""

from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path

# ── Estado de sesion de grabacion ─────────────────────────────────────────────
log_lines: list[str] = []
log_lock = threading.Lock()
cmd_queue: queue.Queue[str] = queue.Queue()
running = False
ep_current = 0
ep_total = 50
status = "idle"  # idle | waiting | recording | done | error
live_metrics = {"fps": 0.0, "latency_ms": 0.0, "size_mb": 0.0}
last_dataset_root: Path | None = None
last_repo_id: str | None = None

# ── Estado del visualizador / curacion ────────────────────────────────────────
viz_dataset_path: Path | None = None
viz_info: dict = {}
viz_ep_rows: list[dict] = []
viz_video_keys: list[str] = []
viz_tmp_dir: Path = Path(tempfile.mkdtemp(prefix="icub_viz_"))

RESERVED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
curation_marked_delete: set[int] = set()

curation_running = False
curation_status = "idle"  # idle | running | done | error
curation_log_lines: list[str] = []
curation_log_lock = threading.Lock()
curation_last_output: Path | None = None
curation_last_repo_id: str | None = None

# ── Estado de subida a Hugging Face Hub ───────────────────────────────────────
push_running = False
push_status = "idle"  # idle | uploading | done | error
push_last_repo_id: str | None = None  # repo subido exitosamente (para generar el link)

# ── Etiquetas de estado compartidas por los pollers ───────────────────────────
STATUS_LABEL = {
    "idle": "Idle",
    "waiting": "Waiting for START",
    "recording": "Recording",
    "done": "Done",
    "error": "Error",
}
