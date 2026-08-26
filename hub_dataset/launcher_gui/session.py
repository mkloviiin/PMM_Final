"""Control generico de una sesion de grabacion.

No depende de que robot esta grabando: cualquier backend de robot (ver
robots/base.py) le pasa a `start_session` una funcion `record_fn(cmd_source,
on_status)` que hace el trabajo real (VR, control remoto, lo que sea) y
termina escribiendo el dataset en formato LeRobot bajo `dataset_root`. Este
modulo se limita a manejar el thread, el log, el estado y el protocolo de
comandos -- por eso mismo "Control de episodios" en la UI no cambia nunca
sin importar el robot activo.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable
import gradio as gr


from . import state


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    with state.log_lock:
        state.log_lines.append(f"[{ts}] {msg}")
        if len(state.log_lines) > 600:
            state.log_lines.pop(0)


class _Tee:
    """Redirige print() al log y al stdout original."""

    def __init__(self, orig):
        self._orig = orig

    def write(self, s: str) -> int:
        if s.strip():
            log(s.rstrip())
        return self._orig.write(s)

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()


def _cmd_source() -> str | None:
    try:
        return state.cmd_queue.get_nowait()
    except queue.Empty:
        return None


def _on_status(event: str | dict) -> None:
    if isinstance(event, dict):
        if "status" in event:
            state.status = event["status"]
        if "metrics" in event:
            state.live_metrics.update(event["metrics"])
    else:
        if event == "waiting":
            state.status = "waiting"
        elif event == "recording":
            state.status = "recording"
        elif event.startswith("saved:"):
            state.ep_current = int(event.split(":")[1])
            state.status = "waiting"


def cmd_close_window() -> str:
    """Cierra la ventana MuJoCo y termina la sesion activa inmediatamente."""
    if state.running:
        state.cmd_queue.put("3")
        return "Closing window and session..."
    return "No active session."


def start_session(record_fn: Callable[..., None], *, repo_id: str,
                   dataset_root: Path, num_episodes: int) -> str:
    """Arranca `record_fn(cmd_source=..., on_status=...)` en un thread propio.

    Si ya hay una sesion activa, le envia EXIT y espera hasta 15 s a que
    termine antes de lanzar la nueva. Esto permite relanzar con otra escena
    sin necesidad de terminar manualmente la sesion anterior.

    `record_fn` es responsabilidad exclusiva del backend del robot: recibe los
    callbacks del protocolo de sesion y debe dejar el dataset final en formato
    LeRobot bajo `dataset_root`. Devuelve el mensaje de estado a mostrar en el
    launcher.
    """
    if state.running:
        # Terminar sesion activa antes de lanzar la nueva
        log("[Hub] Previous session active — sending EXIT to relaunch with new scene.")
        state.cmd_queue.put("3")
        deadline = time.time() + 15
        while state.running and time.time() < deadline:
            time.sleep(0.1)
        if state.running:
            return "Could not close the previous session in time. Please retry in a few seconds."

    while not state.cmd_queue.empty():
        state.cmd_queue.get_nowait()
    with state.log_lock:
        state.log_lines.clear()

    state.ep_total = num_episodes
    state.ep_current = 0
    state.running = True

    def _thread_body() -> None:
        orig = sys.stdout
        sys.stdout = _Tee(orig)
        try:
            state.status = "waiting"
            log(f"Session started - repo: {repo_id}")
            log(f"  dataset: {dataset_root}")
            log(f"  conda env: {os.environ.get('CONDA_DEFAULT_ENV', '?')}  ({sys.executable})")
            record_fn(cmd_source=_cmd_source, on_status=_on_status)
            state.status = "done"
            state.last_dataset_root = dataset_root
            state.last_repo_id = repo_id
            log("Session ended correctly.")
        except Exception:
            log("Error in session:")
            log(traceback.format_exc())
            state.status = "error"
        finally:
            sys.stdout = orig
            state.running = False

    threading.Thread(target=_thread_body, daemon=True).start()
    return "Launching... waiting for robot to connect (~10-20 s)"


def cmd_start() -> str:
    if state.running:
        state.cmd_queue.put("1")
        return "Starting episode..."
    return "No active session."


def cmd_stop() -> str:
    if state.running:
        state.cmd_queue.put("2")
        return "Stopping episode..."
    return "No active session."


def cmd_exit() -> str:
    if state.running:
        state.cmd_queue.put("3")
        return "Closing session..."
    return "No active session."


def poll_status():
    label = "" if state.status == "idle" else state.STATUS_LABEL.get(state.status, state.status)
    pct = state.ep_current / max(state.ep_total, 1)
    progress = f"Episode: {state.ep_current} / {state.ep_total}  ({int(pct * 100)}%)"

    with state.log_lock:
        log_text = "\n".join(state.log_lines[-80:])

    return (
        label, progress, state.ep_current, log_text,
        state.live_metrics["fps"], state.live_metrics["latency_ms"], state.live_metrics["size_mb"],
        gr.update(interactive=not state.running)
    )
