"""Pestana 'Visualizar y curar': agnostica al robot -- solo lee/escribe
datasets en formato LeRobot, sin saber nunca como se grabaron.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import state


def log_curation(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    with state.curation_log_lock:
        state.curation_log_lines.append(f"[{ts}] {msg}")
        if len(state.curation_log_lines) > 400:
            state.curation_log_lines.pop(0)


# ── Descubrimiento de datasets ────────────────────────────────────────────────

def _scan_datasets(root_dir: str) -> list[str]:
    root = Path(root_dir).expanduser()
    if not root.exists():
        return []
    return [
        str(d) for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if d.is_dir() and (d / "meta" / "info.json").exists()
    ]


def _extract_repo_id(dataset_path: str) -> str:
    path = Path(dataset_path)

    if state.last_dataset_root and state.last_repo_id and path == state.last_dataset_root:
        return state.last_repo_id

    try:
        with open(path / "meta" / "info.json", encoding="utf-8") as f:
            info = json.load(f)
        rid = info.get("repo_id") or info.get("name")
        if rid:
            return rid
    except Exception:
        pass

    name = path.name
    m = re.match(r'^(.+)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', name)
    base = m.group(1) if m else name
    if '_' in base:
        idx = base.index('_')
        return base[:idx] + '/' + base[idx + 1:]
    return base


def refresh_datasets(root_dir: str):
    datasets = _scan_datasets(root_dir)
    if not datasets:
        return gr.update(choices=[], value=None), "No datasets found in that folder."
    return gr.update(choices=datasets, value=datasets[0]), f"{len(datasets)} dataset(s) found."


def use_last_dataset(root_dir: str):
    if state.last_dataset_root and state.last_dataset_root.exists():
        return gr.update(value=str(state.last_dataset_root))
    datasets = _scan_datasets(root_dir)
    if datasets:
        return gr.update(value=datasets[0])
    return gr.update(value="")


# ── Visualizacion de episodios ────────────────────────────────────────────────

def _trim_episode_video(video_key: str, ep_row: dict) -> str | None:
    """Recorta el segmento del episodio del video compartido usando ffmpeg."""
    if state.viz_dataset_path is None:
        return None

    chunk_idx = int(ep_row.get(f"videos/{video_key}/chunk_index", 0))
    file_idx = int(ep_row.get(f"videos/{video_key}/file_index", 0))
    t_start = float(ep_row.get(f"videos/{video_key}/from_timestamp", 0))
    t_end = float(ep_row.get(f"videos/{video_key}/to_timestamp", 0))

    src = (state.viz_dataset_path / "videos" / video_key
           / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4")
    if not src.exists():
        return None

    ep_idx = int(ep_row["episode_index"])
    safe_key = video_key.replace("/", "_").replace(".", "_")
    dst = state.viz_tmp_dir / f"ep{ep_idx:04d}_{safe_key}.mp4"

    if not dst.exists():
        duration = max(t_end - t_start, 0.1)
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(t_start), "-i", str(src),
             "-t", str(duration), "-c", "copy", str(dst)],
            capture_output=True,
        )

    return str(dst) if dst.exists() else None


    def _line_chart(values_list: list[np.ndarray], names: list[str], title: str) -> go.Figure:
        fig = go.Figure()
        for i, (values, name) in enumerate(zip(values_list, names)):
            fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name=name))
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=10, t=60, b=40),
            height=420,
        )
        return fig

    return None

def build_custom_plotly_chart(ep_idx: int, variables: list[str], title: str = "") -> go.Figure | None:
    if state.viz_dataset_path is None or not variables or not state.viz_ep_rows:
        return None

    if ep_idx < 0 or ep_idx >= len(state.viz_ep_rows):
        return None

    ep_row = state.viz_ep_rows[ep_idx]
    chunk_idx = int(ep_row["data/chunk_index"])
    file_idx = int(ep_row["data/file_index"])

    parquet = (state.viz_dataset_path / "data"
               / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet")
    if not parquet.exists():
        return None

    df = pd.read_parquet(parquet)
    ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    if ep_df.empty:
        return None

    timestamps = ep_df["timestamp"].astype(float).values
    
    values_list = []
    names = []
    
    action_names = state.viz_info.get("features", {}).get("action", {}).get("names") or []
    state_names = state.viz_info.get("features", {}).get("observation.state", {}).get("names") or []

    for var in variables:
        if var.startswith("action/"):
            var_name = var.split("action/")[1]
            if "action" in ep_df.columns and var_name in action_names:
                idx = action_names.index(var_name)
                values_list.append(np.stack(ep_df["action"].values)[:, idx])
                names.append(var)
        elif var.startswith("observation.state/"):
            var_name = var.split("observation.state/")[1]
            if "observation.state" in ep_df.columns and var_name in state_names:
                idx = state_names.index(var_name)
                values_list.append(np.stack(ep_df["observation.state"].values)[:, idx])
                names.append(var)

    if not values_list:
        return None

    fig = go.Figure()
    for values, name in zip(values_list, names):
        fig.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name=name))
    display_title = title if title else f"Custom Plot — Episode {ep_idx}"
    fig.update_layout(
        title=display_title,
        xaxis_title="Time (s)",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=10, t=60, b=40),
        height=420,
        template="plotly_dark"
    )
    return fig


def _ep_info_str(ep_row: dict) -> str:
    tasks = ep_row.get("tasks", [])
    if hasattr(tasks, "tolist"):
        tasks = tasks.tolist()
    task_str = " / ".join(str(t) for t in tasks) if tasks else "—"
    n_frames = int(ep_row.get("length", 0))
    fps = state.viz_info.get("fps", 30)
    duration = n_frames / fps if fps else 0
    ep_idx = int(ep_row["episode_index"])
    return f"Episode {ep_idx}  |  {n_frames} frames  |  {duration:.1f} s  |  Task: {task_str}"


def _marked_summary() -> str:
    if not state.curation_marked_delete:
        return "Excluded episodes: none."
    if state.viz_ep_rows and len(state.curation_marked_delete) == len(state.viz_ep_rows):
        return "Excluded episodes: all."
    excluded = ", ".join(str(i) for i in sorted(state.curation_marked_delete))
    return f"Excluded episodes: {excluded}"


def _goto_episode(idx):
    """Devuelve todos los outputs de la vista de un episodio (numero clamped,
    info, videos, estado del boton eliminar y resumen de marcados)."""
    if not state.viz_ep_rows or state.viz_dataset_path is None:
        return (0, "No dataset loaded.", None, None, None,
                gr.update(value="🗑️ Delete this episode"), "Excluded episodes: none.")

    ep_idx = min(max(int(idx), 0), len(state.viz_ep_rows) - 1)
    ep_row = state.viz_ep_rows[ep_idx]

    cam_keys = (state.viz_video_keys + [None, None, None])[:3]
    v0, v1, v2 = [_trim_episode_video(k, ep_row) if k else None for k in cam_keys]

    del_label = "↩️ Undo delete" if ep_idx in state.curation_marked_delete else "🗑️ Delete this episode"
    return ep_idx, _ep_info_str(ep_row), v0, v1, v2, gr.update(value=del_label), _marked_summary()


def ep_goto(idx):
    return _goto_episode(idx)


def ep_prev(idx):
    return _goto_episode(int(idx) - 1)


def ep_next(idx):
    return _goto_episode(int(idx) + 1)


def toggle_delete_episode(idx):
    ep_idx = int(idx)
    if ep_idx in state.curation_marked_delete:
        state.curation_marked_delete.discard(ep_idx)
        label = "🗑️ Delete this episode"
    else:
        state.curation_marked_delete.add(ep_idx)
        label = "↩️ Undo delete"
    return gr.update(value=label), _marked_summary()


def load_viz_dataset(dataset_path: str):
    """Carga un dataset y resetea todo el estado de curacion (episodios marcados,
    edicion de tarea pendiente, packaging anterior y estado de subida) asociado
    a la carga anterior. El token de HF se preserva intencionalmente."""
    # Resetear estado de packaging/upload de la carga anterior
    state.curation_last_output = None
    state.curation_status = "idle"
    state.push_status = "idle"
    state.push_last_repo_id = None

    _empty_nav = (
        0, "", None, None, None,
        gr.update(value="🗑️ Delete this episode"), "Excluded episodes: none.",
    )
    _empty_curation = (
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(value="", interactive=False),
        gr.update(interactive=True), gr.update(interactive=False),
        False,
        gr.update(value=""),
        gr.update(choices=[]), # new_plot_vars
        [], # dynamic_plots_state
        gr.update(value=""), # new_plot_title
        # ── Packaging & upload reset ──
        gr.update(value=""),       # package_status
        gr.update(value=""),       # curation_log_box
        gr.update(value=""),       # push_status
        gr.update(value=""),       # push_link_html
        gr.update(value=""),       # hf_repo_id_tb
    )

    if not dataset_path or not dataset_path.strip():
        return (gr.update(), "Select a dataset first.") + _empty_nav + _empty_curation
    path = Path(dataset_path.strip())
    if not (path / "meta" / "info.json").exists():
        return (gr.update(), f"Invalid dataset — could not find meta/info.json in: {path}") + _empty_nav + _empty_curation

    try:
        state.viz_info = json.loads((path / "meta" / "info.json").read_text())

        ep_files = sorted((path / "meta" / "episodes").rglob("*.parquet"))
        if not ep_files:
            return (gr.update(), "Invalid dataset — no episode files found (meta/episodes/*.parquet).") + _empty_nav + _empty_curation

        df = (
            pd.concat([pd.read_parquet(f) for f in ep_files])
            .sort_values("episode_index")
            .reset_index(drop=True)
        )
        state.viz_ep_rows = df.to_dict("records")
        state.viz_video_keys = [
            k for k, v in state.viz_info.get("features", {}).items()
            if v.get("dtype") == "video"
        ]
        state.viz_dataset_path = path
        state.curation_marked_delete = set()

        n_ep = len(state.viz_ep_rows)
        fps = state.viz_info.get("fps", 30)
        total_frames = state.viz_info.get("total_frames",
                                           sum(int(r.get("length", 0)) for r in state.viz_ep_rows))
        status = (f"Dataset loaded | {n_ep} episode(s) | "
                  f"{total_frames} frames | {fps} FPS | "
                  f"{len(state.viz_video_keys)} camera(s)")

        nav = _goto_episode(0)

        features = state.viz_info.get("features", {})
        whole_choices = [k for k in features if k not in state.RESERVED_FEATURES]
        action_names = features.get("action", {}).get("names") or []
        state_names = features.get("observation.state", {}).get("names") or []
        first_task = (state.viz_ep_rows[0].get("tasks") or ["-"])[0]
        suggested_repo_id = f"{_extract_repo_id(dataset_path)}_curated"

        all_plotable_vars = [f"action/{n}" for n in action_names] + [f"observation.state/{n}" for n in state_names]

        curation = (
            gr.update(choices=whole_choices, value=list(whole_choices)),
            gr.update(choices=action_names, value=list(action_names)),
            gr.update(choices=state_names, value=list(state_names)),
            gr.update(value=first_task, interactive=False),
            gr.update(interactive=True), gr.update(interactive=False),
            False,
            gr.update(value=suggested_repo_id),
            gr.update(choices=all_plotable_vars, value=[]),
            [],
            gr.update(value=""),
            # ── Packaging & upload reset ──
            gr.update(value=""),                    # package_status
            gr.update(value=""),                    # curation_log_box
            gr.update(value=""),                # push_status
            gr.update(value=""),                    # push_link_html
            gr.update(value=suggested_repo_id),     # hf_repo_id_tb (sugerencia)
        )

        return (gr.update(visible=True), status) + nav + curation
    except Exception as e:
        state.viz_dataset_path = None
        return (gr.update(), f"Invalid dataset — error while loading: {e}") + _empty_nav + _empty_curation


# ── Search filters for signal lists ──────────────────────────────────────────

def filter_whole_features(query: str, current_value: list) -> dict:
    all_choices = [k for k in state.viz_info.get("features", {}) if k not in state.RESERVED_FEATURES]
    if not query or not query.strip():
        return gr.update(choices=all_choices)
    q = query.strip().lower()
    kept = [c for c in (current_value or []) if c in all_choices]
    matched = [c for c in all_choices if q in c.lower() and c not in kept]
    return gr.update(choices=kept + matched)


def filter_action_dims(query: str, current_value: list) -> dict:
    all_choices = list(state.viz_info.get("features", {}).get("action", {}).get("names") or [])
    if not query or not query.strip():
        return gr.update(choices=all_choices)
    q = query.strip().lower()
    kept = [c for c in (current_value or []) if c in all_choices]
    matched = [c for c in all_choices if q in c.lower() and c not in kept]
    return gr.update(choices=kept + matched)


def filter_state_dims(query: str, current_value: list) -> dict:
    all_choices = list(state.viz_info.get("features", {}).get("observation.state", {}).get("names") or [])
    if not query or not query.strip():
        return gr.update(choices=all_choices)
    q = query.strip().lower()
    kept = [c for c in (current_value or []) if c in all_choices]
    matched = [c for c in all_choices if q in c.lower() and c not in kept]
    return gr.update(choices=kept + matched)


# ── Select all/none helpers ───────────────────────────────────────────────────

def select_all_whole():
    return gr.update(value=[k for k in state.viz_info.get("features", {}) if k not in state.RESERVED_FEATURES])


def select_all_action_dims():
    return gr.update(value=list(state.viz_info.get("features", {}).get("action", {}).get("names") or []))


def select_all_state_dims():
    return gr.update(value=list(state.viz_info.get("features", {}).get("observation.state", {}).get("names") or []))


def select_none_list():
    return gr.update(value=[])


# ── Edicion de la instruccion global ──────────────────────────────────────────

def task_start_edit(current_text):
    """Activa el modo edicion del textbox de instruccion.
    Habilita el textbox y el boton guardar, deshabilita el boton editar.
    Ambos botones permanecen siempre en el DOM (se controla con interactive,
    no con visible) para evitar problemas de layout en Gradio.
    """
    return (
        gr.update(value=current_text, interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def task_save(text):
    """Guarda el texto, deshabilita el textbox, restaura el boton editar."""
    return (
        gr.update(value=text, interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        True,
    )


# ── Empaquetado ────────────────────────────────────────────────────────────────

def run_packaging(dataset_path, whole_kept, action_kept, state_kept,
                   task_text, task_dirty, new_repo_id, curated_root_dir):
    if state.curation_running:
        return gr.update(value="A packaging process is already running.")
    if state.viz_dataset_path is None or not dataset_path:
        return gr.update(value="Load a dataset first.")

    rid = (new_repo_id.strip() or f"{_extract_repo_id(dataset_path)}_curated")
    rid = rid if "/" in rid else f"local/{rid}"

    features = state.viz_info.get("features", {})
    all_whole = [k for k in features if k not in state.RESERVED_FEATURES]
    whole_features_to_drop = [k for k in all_whole if k not in (whole_kept or [])]

    dims_to_drop: dict[str, list[str]] = {}

    all_action_names = features.get("action", {}).get("names") or []
    if all_action_names and "action" not in whole_features_to_drop:
        if not action_kept:
            whole_features_to_drop.append("action")
        else:
            to_drop = [n for n in all_action_names if n not in action_kept]
            if to_drop:
                dims_to_drop["action"] = to_drop

    all_state_names = features.get("observation.state", {}).get("names") or []
    if all_state_names and "observation.state" not in whole_features_to_drop:
        if not state_kept:
            whole_features_to_drop.append("observation.state")
        else:
            to_drop = [n for n in all_state_names if n not in state_kept]
            if to_drop:
                dims_to_drop["observation.state"] = to_drop

    episodes_drop = sorted(state.curation_marked_delete)
    task_global = task_text.strip() if (task_dirty and task_text and task_text.strip()) else None

    run_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_root = Path(curated_root_dir).expanduser() / f"{rid.replace('/', '_')}_{run_suffix}"

    state.curation_running = True
    with state.curation_log_lock:
        state.curation_log_lines.clear()

    threading.Thread(
        target=_run_packaging_thread,
        args=(
            Path(dataset_path.strip()), _extract_repo_id(dataset_path),
            episodes_drop, whole_features_to_drop, dims_to_drop,
            task_global, None, rid, final_root,
        ),
        daemon=True,
    ).start()

    return gr.update(value=f"Packaging in background -> {final_root}")


def _run_packaging_thread(src_root, src_repo_id, episodes_drop, whole_features,
                           dims_to_drop, task_global, episode_tasks, rid, final_root):
    from dependencies.dataset_curation import package_dataset

    try:
        state.curation_status = "running"
        final_path = package_dataset(
            source_root=src_root,
            source_repo_id=src_repo_id,
            episodes_to_drop=episodes_drop or None,
            whole_features_to_drop=whole_features or None,
            dims_to_drop=dims_to_drop or None,
            task_global=task_global,
            episode_tasks=episode_tasks,
            final_repo_id=rid,
            final_root=final_root,
            on_log=log_curation,
        )
        state.curation_last_output = final_path
        state.curation_last_repo_id = rid
        state.curation_status = "done"
        log_curation(f"Packaging complete: {final_path}")
    except Exception:
        state.curation_status = "error"
        log_curation(traceback.format_exc())
    finally:
        state.curation_running = False


def poll_curation_status():
    with state.curation_log_lock:
        log_text = "\n".join(state.curation_log_lines[-60:])
    curation_status = state.curation_status
    status_text = "" if curation_status == "idle" else state.STATUS_LABEL.get(curation_status, curation_status)
    return status_text, log_text
