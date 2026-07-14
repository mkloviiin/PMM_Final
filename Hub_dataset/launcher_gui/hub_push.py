"""Pestana 'Subir a Hugging Face Hub': agnostica al robot -- opera sobre el
dataset ya empaquetado (formato LeRobot) por la pestana de curacion.
"""

from __future__ import annotations

import os
import threading
import traceback

import gradio as gr

from . import state
from .curation import log_curation


def push_to_hub_action(hf_repo_id: str, hf_token: str, private: bool):
    if state.curation_last_output is None:
        return gr.update(value="Primero empaqueta un dataset en esta pestaña.")

    hf_repo_id = (hf_repo_id or "").strip()
    if hf_repo_id.startswith("local/"):
        return gr.update(value="Usa tu usuario/organizacion real de Hugging Face (no 'local/...').")
    if "/" not in hf_repo_id:
        return gr.update(value="Repo ID invalido. Formato: usuario/nombre_dataset")
    if not (hf_token or "").strip():
        return gr.update(value="Ingresa tu token de Hugging Face antes de subir.")
    if state.push_running:
        return gr.update(value="Ya hay una subida en curso.")

    state.push_running = True
    threading.Thread(
        target=_run_push_thread,
        args=(state.curation_last_output, hf_repo_id, hf_token.strip(), private),
        daemon=True,
    ).start()
    return gr.update(value=f"Subiendo {state.curation_last_output} a {hf_repo_id}...")


def _run_push_thread(root, hf_repo_id: str, token: str, private: bool):
    prev_token = os.environ.get("HF_TOKEN")
    try:
        state.push_status = "uploading"
        os.environ["HF_TOKEN"] = token
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id=hf_repo_id, root=root)
        ds.push_to_hub(private=private)
        state.push_status = "done"
        log_curation(f"Subido: https://huggingface.co/datasets/{hf_repo_id}")
    except Exception:
        state.push_status = "error"
        log_curation(traceback.format_exc())
    finally:
        if prev_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = prev_token
        state.push_running = False


def poll_push_status():
    return gr.update(value=state.STATUS_LABEL.get(state.push_status, state.push_status))
