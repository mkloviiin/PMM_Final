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


from . import session_log, state


# Log completo en disco, uno por ejecución del Hub: la UI solo muestra las últimas
# líneas. Se escribe línea a línea (no al cerrar) para no perderlo si el Hub se cae.
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / f"hub_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
_log_fh = None


def _log_to_file(line: str) -> None:
    """Agrega una línea al archivo de log (llamar con state.log_lock tomado)."""
    global _log_fh
    try:
        if _log_fh is None:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _log_fh = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        _log_fh.write(line + "\n")
    except Exception:  # noqa: BLE001 -- el log en disco nunca debe tumbar el Hub
        pass


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with state.log_lock:
        state.log_lines.append(line)
        if len(state.log_lines) > 600:
            state.log_lines.pop(0)
        _log_to_file(line)


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


def _set_status(new: str) -> None:
    if new == "recording" and state.status != "recording":
        session_log.on_recording_start()  # empieza un episodio: nuevas muestras de métricas
    state.status = new


def _on_status(event: str | dict) -> None:
    if isinstance(event, dict):
        if "status" in event:
            _set_status(event["status"])
        if "metrics" in event:
            state.live_metrics.update(event["metrics"])
            session_log.on_metrics(event["metrics"], recording=state.status == "recording")
    else:
        if event == "waiting":
            _set_status("waiting")
        elif event == "recording":
            _set_status("recording")
        elif event.startswith("saved:"):
            state.ep_current = int(event.split(":")[1])
            session_log.on_saved(state.ep_current, state.live_metrics.get("size_mb", 0.0))
            _set_status("waiting")


def cmd_close_window() -> str:
    """Cierra la ventana MuJoCo y termina la sesion activa inmediatamente."""
    if state.running:
        state.cmd_queue.put("3")
        return "Closing window and session..."
    return "No active session."


def start_session(record_fn: Callable[..., None], *, repo_id: str,
                   dataset_root: Path, num_episodes: int, vr: bool = False) -> str:
    """Arranca `record_fn(cmd_source=..., on_status=...)` en un thread propio.

    Si ya hay una sesion activa, le envia EXIT y espera hasta 15 s a que
    termine antes de lanzar la nueva. Esto permite relanzar con otra escena
    sin necesidad de terminar manualmente la sesion anterior.

    `record_fn` es responsabilidad exclusiva del backend del robot: recibe los
    callbacks del protocolo de sesion y debe dejar el dataset final en formato
    LeRobot bajo `dataset_root`. `vr`: la sesión usa VR (la vista en vivo de Episode
    Control muestra lo que va a los lentes). Devuelve el mensaje de estado a mostrar en el
    launcher.
    """
    if state.running:
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
        _log_to_file(f"\n===== New session: {repo_id} ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====")
    log(f"[Hub] Full log: {LOG_FILE}")

    state.ep_total = num_episodes
    state.ep_current = 0
    state.running = True
    state.live_view_vr = bool(vr)
    state.session_seq += 1
    session_log.begin(dataset_root)  # métricas por episodio + bitácora de esta sesión

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


def set_planned_episodes(num_episodes) -> None:
    """'Number of episodes' del formulario -> total de la barra de progreso (sin sesión activa)."""
    if not state.running and num_episodes:
        state.ep_total = max(int(num_episodes), 1)


def reset_episode_control(num_episodes) -> bool:
    """Deja Episode Control como nuevo (estado, progreso, log, métricas, gráfico por episodio y
    bitácora), p. ej. al cambiar de plataforma después de haber grabado con otra. No hace nada
    con una sesión activa. "Use last recorded" (Visualize & Curate) se conserva: ese dataset
    sigue existiendo y curar no depende de la plataforma."""
    if state.running:
        return False
    state.status = "idle"
    state.ep_current = 0
    set_planned_episodes(num_episodes)
    with state.log_lock:
        state.log_lines.clear()
    state.live_metrics.update(fps=0.0, latency_ms=0.0, size_mb=0.0)
    session_log.reset()
    return True


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


def _episode_metrics_figure():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = session_log.rows()
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "FPS (mean ± std)", "Latency [ms] (mean ± std)", "Episode duration [s]", "Dataset size [MB]"))
    ep = [r["episode"] for r in rows]
    fig.add_trace(go.Scatter(x=ep, y=[r["fps_mean"] for r in rows], mode="lines+markers", name="FPS",
                             error_y=dict(type="data", array=[r["fps_std"] for r in rows])), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=[r["latency_ms_mean"] for r in rows], mode="lines+markers", name="Latency",
                             error_y=dict(type="data", array=[r["latency_ms_std"] for r in rows])), 1, 2)
    fig.add_trace(go.Bar(x=ep, y=[r["duration_s"] for r in rows], name="Duration"), 2, 1)
    fig.add_trace(go.Scatter(x=ep, y=[r["dataset_size_mb"] for r in rows], mode="lines+markers", name="Size"), 2, 2)
    fig.update_xaxes(tickformat="d", rangemode="tozero")  # episodios: solo enteros
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_layout(height=520, showlegend=False, template="plotly_dark",
                      margin=dict(l=40, r=20, t=50, b=40),
                      title=("Per-episode metrics" if rows else "Per-episode metrics — no saved episodes yet"))
    return fig


def _notes_markdown() -> str:
    notes = session_log.notes()
    if not notes:
        return "*No notes yet.*"
    lines = ["| Time | Episode | Phase | Note |", "|---|---|---|---|"]
    for n in reversed(notes):
        note = n["note"].replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {n['time'][11:]} | {n['episode']} | {n['phase']} | {note} |")
    folder = session_log.folder()
    if folder is not None:
        lines.append(f"\nSaved in `{folder}`")
    return "\n".join(lines)


def episode_panels():
    """Gráfico por episodio + notas, completos (al abrir la pestaña Episode Control: las
    actualizaciones del poller no llegan a componentes que aún no están dibujados)."""
    return _episode_metrics_figure(), _notes_markdown()


def add_note(text: str):
    """Botón 'Add note' de la bitácora: asocia la nota al episodio en curso / siguiente."""
    if not (text or "").strip():
        return gr.update(), gr.update()
    if not session_log.has_session():
        gr.Warning("Launch a session first: the logbook is saved in the dataset folder.")
        return gr.update(), gr.update()
    if state.status == "recording":
        episode, phase = state.ep_current, "recording"
    elif state.running:
        episode, phase = state.ep_current, "before start"
    else:
        episode, phase = max(state.ep_current - 1, 0), "after session"
    session_log.add_note(text, episode, phase)
    return "", _notes_markdown()
