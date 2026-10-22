"""Pestana 'Subir a Hugging Face Hub': agnostica al robot. Sube el dataset empaquetado
en la pestana de curacion si lo hay; si no, el dataset cargado tal como se grabo (siempre
que no haya cambios de curacion sin empaquetar). En ambos casos solo el formato LeRobot
(data/, meta/, videos/): hub_session/ (metricas y bitacora del Hub) no se sube.
"""

from __future__ import annotations

import os
import threading
import traceback

import gradio as gr

from . import state
from .curation import log_curation

# Lo que se sube de la carpeta del dataset (images/ ya lo excluye LeRobot)
UPLOAD_PATTERNS = ["data/*", "meta/*", "videos/*"]


def _upload_source():
    """(carpeta, descripción) de lo que subiría el botón, o (None, None)."""
    if state.curation_last_output is not None:
        return state.curation_last_output, "packaged dataset"
    if state.viz_dataset_path is not None:
        return state.viz_dataset_path, "original dataset (as recorded)"
    return None, None


def _pending_curation_changes(whole_kept, action_kept, state_kept, task_dirty) -> list[str]:
    """Cambios de curación hechos en la pestaña que el dataset original no tiene."""
    features = state.viz_info.get("features", {})
    changes = []
    if state.curation_marked_delete:
        changes.append(f"{len(state.curation_marked_delete)} episode(s) marked for deletion")
    all_whole = [k for k in features if k not in state.RESERVED_FEATURES]
    if set(whole_kept or []) != set(all_whole):
        changes.append("features deselected")
    for key, kept in (("action", action_kept), ("observation.state", state_kept)):
        names = features.get(key, {}).get("names") or []
        if names and set(kept or []) != set(names):
            changes.append(f"{key} dimensions deselected")
    if task_dirty:
        changes.append("language instruction edited")
    return changes


def push_to_hub_action(hf_repo_id: str, hf_token: str, private: bool,
                       whole_kept=None, action_kept=None, state_kept=None, task_dirty=False):
    root, _ = _upload_source()
    if root is None:
        return gr.update(value="Load a dataset first (section 1).")
    if state.curation_last_output is None:
        changes = _pending_curation_changes(whole_kept, action_kept, state_kept, task_dirty)
        if changes:
            return gr.update(value="You have curation changes that aren't packaged ("
                                   + "; ".join(changes) + "). Package the dataset first "
                                   "(section 6) or undo them to upload the original.")

    hf_repo_id = (hf_repo_id or "").strip()
    if hf_repo_id.startswith("local/"):
        return gr.update(value="Use your real Hugging Face user/organization (not 'local/...').")
    if "/" not in hf_repo_id:
        return gr.update(value="Invalid Repo ID. Format: user/dataset_name")
    if not (hf_token or "").strip():
        return gr.update(value="Enter your Hugging Face token before uploading.")
    if state.push_running:
        return gr.update(value="An upload is already in progress.")

    state.push_running = True
    threading.Thread(
        target=_run_push_thread,
        args=(root, hf_repo_id, hf_token.strip(), private),
        daemon=True,
    ).start()
    return gr.update(value=f"Uploading {root} to {hf_repo_id}...")


def _run_push_thread(root, hf_repo_id: str, token: str, private: bool):
    prev_token = os.environ.get("HF_TOKEN")
    try:
        state.push_status = "uploading"
        os.environ["HF_TOKEN"] = token
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id=hf_repo_id, root=root)
        ds.push_to_hub(private=private, allow_patterns=UPLOAD_PATTERNS)
        state.push_last_repo_id = hf_repo_id
        state.push_status = "done"
        log_curation(f"Uploaded {root} -> https://huggingface.co/datasets/{hf_repo_id}")
    except Exception:
        state.push_status = "error"
        log_curation(traceback.format_exc())
    finally:
        if prev_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = prev_token
        state.push_running = False


def _upload_target_markdown() -> str:
    root, kind = _upload_source()
    if root is None:
        return "*Will upload: nothing yet — load a dataset in section 1.*"
    return f"**Will upload:** {kind} — `{root}`"


def poll_push_status():
    status = state.push_status
    status_text = "" if status == "idle" else state.STATUS_LABEL.get(status, status)
    if state.push_status == "done" and state.push_last_repo_id:
        url = f"https://huggingface.co/datasets/{state.push_last_repo_id}"
        link_html = (
            f'<div style="display:flex;justify-content:center;width:100%;">'
            f'<a href="{url}" target="_blank" style="'
            f'display:inline-flex;align-items:center;justify-content:center;gap:6px;'
            f'width:100%;padding:10px 16px;border-radius:8px;'
            f'background:linear-gradient(135deg,#ff6f00,#ff9800);'
            f'color:#fff;font-weight:600;text-decoration:none;'
            f'font-size:0.95rem;box-shadow:0 2px 8px rgba(255,111,0,.35);'
            f'transition:opacity .2s;" '
            f'onmouseover="this.style.opacity=\'0.85\'" '
            f'onmouseout="this.style.opacity=\'1\'"'
            f'>'
            f'&#127760; Open on Hugging Face'
            f'</a>'
            f'</div>'
        )
    else:
        link_html = ""
    return gr.update(value=status_text), gr.update(value=link_html), gr.update(value=_upload_target_markdown())
