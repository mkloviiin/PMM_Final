"""Pestaña Export del Hub v2: empaquetar la copia curada y subirla a Hugging Face.

Lo nuevo frente a las secciones 6-7 de Visualize & Curate: arriba, un resumen de lo que va a
cambiar respecto del dataset grabado (episodios excluidos, señales quitadas, instrucción
editada), para revisar antes de exportar en vez de recordar lo que se marcó en otras pestañas.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from launcher_gui import curation, hub_push, state
from launcher_gui.paths import REPO_ROOT
from launcher_gui.ui.dir_picker import dir_textbox

EXPORT_TAB_ID = "export"

_HF_HELP = (
    "- Create a token with **Write** permission at "
    "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). It is used only "
    "for this upload and never saved.\n"
    "- The repo ID must start with your Hugging Face username or an organization you can write "
    "to (not `local/...`), otherwise the upload fails with 403 Forbidden."
)


def build_export_components() -> dict:
    """Componentes de Export (se crean dentro de su pestaña desde app.py)."""
    e: dict = {}
    e["summary_md"] = gr.Markdown("", elem_classes=["v2-card"])
    with gr.Row(equal_height=False):
        with gr.Column(elem_classes=["v2-card"]):
            gr.Markdown("1 · Save a curated copy (local)", elem_classes=["v2-card-title"])
            e["new_repo_id_tb"] = gr.Textbox(label="Curated dataset name")
            e["curated_root_tb"], _ = dir_textbox("Folder", str(REPO_ROOT / "data_curated"),
                                                  title="Select the folder for curated datasets")
            e["package_btn"] = gr.Button("Save curated copy", variant="primary")
            e["package_status"] = gr.Textbox(label="Status", interactive=False)
            with gr.Accordion("Details", open=False):
                e["curation_log_box"] = gr.Textbox(show_label=False, lines=8, max_lines=8, interactive=False)
        with gr.Column(elem_classes=["v2-card"]):
            gr.Markdown("2 · Publish to Hugging Face (optional)", elem_classes=["v2-card-title"])
            e["upload_target_md"] = gr.Markdown("", elem_classes=["v2-muted"])
            with gr.Row():
                e["hf_repo_id_tb"] = gr.Textbox(label="Repo ID", placeholder="your_user/dataset_name", scale=3)
                e["hf_private_cb"] = gr.Checkbox(value=True, label="Private", scale=1)
            e["hf_token_tb"] = gr.Textbox(label="Token", type="password", placeholder="hf_...")
            with gr.Accordion("How do I get a token?", open=False):
                gr.Markdown(_HF_HELP)
            e["push_btn"] = gr.Button("Upload", variant="stop")
            e["push_status"] = gr.Textbox(label="Status", interactive=False)
            e["push_link_html"] = gr.HTML("")
    return e


def changes_summary(whole_kept, action_kept, state_kept, task_dirty, task_text) -> str:
    if state.viz_dataset_path is None or not state.viz_info:
        return "### Nothing to export yet\nOpen a dataset in **Review & Curate** first."
    feats = state.viz_info.get("features", {})
    n_ep = state.viz_info.get("total_episodes", len(state.viz_ep_rows))
    whole = [k for k in feats if k not in state.RESERVED_FEATURES]
    a_names = feats.get("action", {}).get("names") or []
    s_names = feats.get("observation.state", {}).get("names") or []
    excl = sorted(state.curation_marked_delete)
    rows = []
    if excl:
        rows.append(f"- **Episodes excluded:** {len(excl)} of {n_ep} ({', '.join(map(str, excl[:20]))}"
                    f"{'…' if len(excl) > 20 else ''})")
    dropped = [k for k in whole if k not in (whole_kept or [])]
    if dropped:
        rows.append(f"- **Features removed:** {', '.join(dropped)}")
    for label, names, kept, feat in (("action", a_names, action_kept, "action"),
                                     ("observation.state", s_names, state_kept, "observation.state")):
        if feat in dropped or not names:
            continue
        gone = [n for n in names if n not in (kept or [])]
        if gone:
            rows.append(f"- **{label}:** {len(names) - len(gone)} of {len(names)} dimensions kept "
                        f"(removed: {', '.join(gone[:8])}{'…' if len(gone) > 8 else ''})")
    if task_dirty:
        rows.append(f"- **Instruction:** changed to *“{task_text}”*")
    head = f"### Export of `{Path(state.viz_dataset_path).name}` — {n_ep - len(excl)} episodes\n"
    if not rows:
        return head + "No changes: the curated copy would be identical to the recorded dataset."
    return head + "\n".join(rows)


def wire(e: dict, c: dict) -> None:
    """Eventos de Export; `c` son los componentes de Review & Curate (selección de señales)."""
    e["package_btn"].click(
        curation.run_packaging,
        [c["dataset_dd"], c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"],
         c["task_tb"], c["task_dirty_state"], e["new_repo_id_tb"], e["curated_root_tb"]],
        [e["package_status"]],
    )
    gr.Timer(0.5).tick(curation.poll_curation_status, None, [e["package_status"], e["curation_log_box"]])
    e["push_btn"].click(
        hub_push.push_to_hub_action,
        [e["hf_repo_id_tb"], e["hf_token_tb"], e["hf_private_cb"], c["whole_feature_cbg"],
         c["dims_action_cbg"], c["dims_state_cbg"], c["task_dirty_state"]],
        [e["push_status"]],
    )
    gr.Timer(0.5).tick(hub_push.poll_push_status, None,
                       [e["push_status"], e["push_link_html"], e["upload_target_md"]])
