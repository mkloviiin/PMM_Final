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
        return gr.update(choices=[], value=None), "No se encontraron datasets en esa carpeta."
    return gr.update(choices=datasets, value=datasets[0]), f"{len(datasets)} dataset(s) encontrado(s)."


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


def _build_plotly_charts(ep_row: dict):
    """Lee el parquet del episodio y construye figuras interactivas de accion y estado.

    Plotly renderiza leyendas cliqueables de fabrica: tocar un nombre en la
    leyenda oculta/muestra esa senal, funcionando como un checkbox interactivo
    sin necesitar un componente Gradio custom.
    """
    if state.viz_dataset_path is None:
        return None, None

    chunk_idx = int(ep_row["data/chunk_index"])
    file_idx = int(ep_row["data/file_index"])
    ep_idx = int(ep_row["episode_index"])

    parquet = (state.viz_dataset_path / "data"
               / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet")
    if not parquet.exists():
        return None, None

    df = pd.read_parquet(parquet)
    ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    if ep_df.empty:
        return None, None

    timestamps = ep_df["timestamp"].astype(float).values

    def _line_chart(values: np.ndarray, names: list[str], title: str,
                     default_visible_names: set[str] | None = None) -> go.Figure:
        fig = go.Figure()
        for i, name in enumerate(names):
            visible = True
            if default_visible_names is not None:
                visible = True if name in default_visible_names else "legendonly"
            fig.add_trace(go.Scatter(x=timestamps, y=values[:, i], mode="lines", name=name, visible=visible))
        fig.update_layout(
            title=f"{title} — {len(names)} senal(es), clickea la leyenda para mostrar/ocultar",
            xaxis_title="Tiempo (s)",
            yaxis_title="Valor",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=10, t=60, b=40),
            height=420,
        )
        return fig

    # Grafica de acciones: todas las dimensiones, todas visibles (son pocas)
    fig_action = None
    action_feat = state.viz_info.get("features", {}).get("action", {})
    action_names = action_feat.get("names") or []
    if "action" in ep_df.columns and action_names:
        actions = np.stack(ep_df["action"].values)
        fig_action = _line_chart(actions, action_names, f"Acciones — Episodio {ep_idx}")

    # Grafica de estado: TODAS las dimensiones estan en la leyenda (118 tipicamente),
    # pero solo unas pocas claves (EE + objeto + gripper) arrancan visibles para que
    # el grafico sea legible; el resto se puede activar clickeando su nombre.
    fig_state = None
    state_feat = state.viz_info.get("features", {}).get("observation.state", {})
    state_names = state_feat.get("names") or []
    if "observation.state" in ep_df.columns and state_names:
        states = np.stack(ep_df["observation.state"].values)
        key_terms = ["rh_ee_pos", "rh_gripper", "object_pos"]
        default_visible = {n for n in state_names if any(k in n for k in key_terms)}
        fig_state = _line_chart(
            states, state_names, f"Estado — Episodio {ep_idx}",
            default_visible_names=default_visible or None,
        )

    return fig_action, fig_state


def _ep_info_str(ep_row: dict) -> str:
    tasks = ep_row.get("tasks", [])
    if hasattr(tasks, "tolist"):
        tasks = tasks.tolist()
    task_str = " / ".join(str(t) for t in tasks) if tasks else "—"
    n_frames = int(ep_row.get("length", 0))
    fps = state.viz_info.get("fps", 30)
    duration = n_frames / fps if fps else 0
    ep_idx = int(ep_row["episode_index"])
    return f"Episodio {ep_idx}  |  {n_frames} frames  |  {duration:.1f} s  |  Tarea: {task_str}"


def _marked_summary() -> str:
    if not state.curation_marked_delete:
        return "Episodios no incluidos: ninguno."
    if state.viz_ep_rows and len(state.curation_marked_delete) == len(state.viz_ep_rows):
        return "Episodios no incluidos: todos."
    excluded = ", ".join(str(i) for i in sorted(state.curation_marked_delete))
    return f"Episodios no incluidos: {excluded}"


def _goto_episode(idx):
    """Devuelve todos los outputs de la vista de un episodio (numero clamped,
    info, videos, graficos, estado del boton eliminar y resumen de marcados)."""
    if not state.viz_ep_rows or state.viz_dataset_path is None:
        return (0, "Sin dataset cargado.", None, None, None, None, None,
                gr.update(value="🗑️ Eliminar este episodio"), "Episodios no incluidos: ninguno.")

    ep_idx = min(max(int(idx), 0), len(state.viz_ep_rows) - 1)
    ep_row = state.viz_ep_rows[ep_idx]

    cam_keys = (state.viz_video_keys + [None, None, None])[:3]
    v0, v1, v2 = [_trim_episode_video(k, ep_row) if k else None for k in cam_keys]

    fig_a, fig_s = _build_plotly_charts(ep_row)

    del_label = "↩️ Deshacer eliminacion" if ep_idx in state.curation_marked_delete else "🗑️ Eliminar este episodio"
    return ep_idx, _ep_info_str(ep_row), v0, v1, v2, fig_a, fig_s, gr.update(value=del_label), _marked_summary()


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
        label = "🗑️ Eliminar este episodio"
    else:
        state.curation_marked_delete.add(ep_idx)
        label = "↩️ Deshacer eliminacion"
    return gr.update(value=label), _marked_summary()


def load_viz_dataset(dataset_path: str):
    """Carga un dataset y resetea todo el estado de curacion (episodios marcados,
    edicion de tarea pendiente) asociado a la carga anterior."""
    _empty_nav = (
        0, "", None, None, None, None, None,
        gr.update(value="🗑️ Eliminar este episodio"), "Episodios no incluidos: ninguno.",
    )
    _empty_curation = (
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(choices=[], value=[]),
        gr.update(value="", interactive=False),
        gr.update(visible=True), gr.update(visible=False),
        False,
        gr.update(value=""),
    )

    if not dataset_path or not dataset_path.strip():
        return ("Selecciona un dataset.",) + _empty_nav + _empty_curation
    path = Path(dataset_path.strip())
    if not (path / "meta" / "info.json").exists():
        return (f"No es un dataset LeRobot valido: {path}",) + _empty_nav + _empty_curation

    try:
        state.viz_info = json.loads((path / "meta" / "info.json").read_text())

        ep_files = sorted((path / "meta" / "episodes").rglob("*.parquet"))
        if not ep_files:
            return ("No se encontraron archivos de episodios.",) + _empty_nav + _empty_curation

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
        status = (f"Dataset cargado | {n_ep} episodio(s) | "
                  f"{total_frames} frames | {fps} FPS | "
                  f"{len(state.viz_video_keys)} camara(s)")

        nav = _goto_episode(0)

        features = state.viz_info.get("features", {})
        whole_choices = [k for k in features if k not in state.RESERVED_FEATURES]
        action_names = features.get("action", {}).get("names") or []
        state_names = features.get("observation.state", {}).get("names") or []
        first_task = (state.viz_ep_rows[0].get("tasks") or ["-"])[0]
        suggested_repo_id = f"{_extract_repo_id(dataset_path)}_curated"

        curation = (
            gr.update(choices=whole_choices, value=list(whole_choices)),
            gr.update(choices=action_names, value=list(action_names)),
            gr.update(choices=state_names, value=list(state_names)),
            gr.update(value=first_task, interactive=False),
            gr.update(visible=True), gr.update(visible=False),
            False,
            gr.update(value=suggested_repo_id),
        )

        return (status,) + nav + curation
    except Exception as e:
        state.viz_dataset_path = None
        return (f"Error al cargar: {e}\n{traceback.format_exc()}",) + _empty_nav + _empty_curation


# ── Seleccionar todo/ninguno para las listas de integracion ──────────────────

def select_all_whole():
    return gr.update(value=[k for k in state.viz_info.get("features", {}) if k not in state.RESERVED_FEATURES])


def select_all_action_dims():
    return gr.update(value=list(state.viz_info.get("features", {}).get("action", {}).get("names") or []))


def select_all_state_dims():
    return gr.update(value=list(state.viz_info.get("features", {}).get("observation.state", {}).get("names") or []))


def select_none_list():
    return gr.update(value=[])


# ── Edicion de la instruccion global ──────────────────────────────────────────

def task_start_edit():
    return gr.update(interactive=True), gr.update(visible=False), gr.update(visible=True)


def task_save(text):
    return gr.update(value=text, interactive=False), gr.update(visible=True), gr.update(visible=False), True


# ── Empaquetado ────────────────────────────────────────────────────────────────

def run_packaging(dataset_path, whole_kept, action_kept, state_kept,
                   task_text, task_dirty, new_repo_id, curated_root_dir):
    if state.curation_running:
        return gr.update(value="Ya hay un empaquetado en curso.")
    if state.viz_dataset_path is None or not dataset_path:
        return gr.update(value="Carga un dataset primero.")

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

    return gr.update(value=f"Empaquetando en segundo plano -> {final_root}")


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
        log_curation(f"Empaquetado completo: {final_path}")
    except Exception:
        state.curation_status = "error"
        log_curation(traceback.format_exc())
    finally:
        state.curation_running = False


def poll_curation_status():
    with state.curation_log_lock:
        log_text = "\n".join(state.curation_log_lines[-60:])
    return state.STATUS_LABEL.get(state.curation_status, state.curation_status), log_text
