"""Pestaña Review & Curate del Hub v2.

Mismos componentes y funciones que Visualize & Curate (launcher_gui/curation.py), ordenados de
otra forma:
- Barra de dataset compacta arriba (la carpeta queda plegada: casi nunca se cambia).
- Un estado vacío que dice qué hacer mientras no hay dataset cargado.
- Las 7 secciones de una página larga pasan a sub-pestañas: Episodes · Signal plots · Data to keep
  · Instruction. Empaquetar y subir van en la pestaña Export, con el resumen de cambios.
"""

from __future__ import annotations

import tempfile

import gradio as gr

from launcher_gui import curation, state
from launcher_gui.app import _checkbox_filter_js
from launcher_gui.curation import MODAL_HIDDEN, build_custom_plotly_chart
from launcher_gui.paths import REPO_ROOT
from launcher_gui.ui.dir_picker import dir_textbox

CURATE_TAB_ID = "curate"


def _signal_list(title: str, elem_id: str, placeholder: str):
    with gr.Column(elem_classes=["v2-card"]):
        gr.Markdown(title, elem_classes=["v2-card-title"])
        with gr.Row():
            all_btn = gr.Button("All", size="sm")
            none_btn = gr.Button("None", size="sm")
        search = gr.Textbox(placeholder=placeholder, show_label=False, container=False)
        cbg = gr.CheckboxGroup(choices=[], value=[], show_label=False, elem_classes=["scrollable-list"],
                               elem_id=elem_id)
    return all_btn, none_btn, search, cbg


def build_curate_tab() -> dict:
    """Componentes de la pestaña; los eventos se conectan con wire() una vez creada Export."""
    c: dict = {}
    with gr.Tab("3 · Review & Curate", id=CURATE_TAB_ID, render_children=True) as tab:
        with gr.Row(equal_height=True):
            c["dataset_dd"] = gr.Dropdown(choices=[], value=None, label="Dataset", allow_custom_value=True,
                                          scale=5)
            c["load_ds_btn"] = gr.Button("Open", variant="primary", elem_classes=["btn-green"], scale=1)
            c["last_btn"] = gr.Button("Last recorded", scale=1)
            c["refresh_btn"] = gr.Button("↻", scale=0, min_width=48)
            c["delete_ds_btn"] = gr.Button("🗑", scale=0, min_width=48, elem_classes=["btn-red"])
        with gr.Accordion("Datasets folder", open=False):
            c["viz_root_tb"], c["browse_dir_ev"] = dir_textbox(
                "Datasets folder", str(REPO_ROOT / "data"), title="Select the datasets folder")
        c["viz_status"] = gr.Markdown("", elem_classes=["v2-muted"])

        c["delete_target_state"] = gr.State(None)
        with gr.Column(elem_classes=MODAL_HIDDEN) as c["delete_confirm_grp"]:
            c["delete_confirm_md"] = gr.Markdown("")
            with gr.Row():
                c["delete_cancel_btn"] = gr.Button("Cancel")
                c["delete_yes_btn"] = gr.Button("Delete", variant="primary", elem_classes=["btn-red"])

        c["empty_md"] = gr.Markdown(
            "### No dataset open\nPick one above and press **Open** — or **Last recorded** for the "
            "dataset you just recorded in this Hub.", elem_classes=["v2-card", "v2-empty"])

        with gr.Column(visible=False) as c["dataset_col"]:
            with gr.Tabs():
                # ── Episodios ────────────────────────────────────────────────
                with gr.Tab("Episodes", render_children=True):
                    with gr.Row(elem_classes=["ep-nav-row"]):
                        c["ep_prev_btn"] = gr.Button("◄ Prev", scale=0, min_width=120)
                        c["ep_num_nb"] = gr.Number(value=0, show_label=False, precision=0, scale=0,
                                                   min_width=90, container=False, minimum=0)
                        c["ep_next_btn"] = gr.Button("Next ►", scale=0, min_width=120)
                        c["ep_delete_btn"] = gr.Button("🗑️ Delete this episode", scale=1)
                    c["ep_info_tb"] = gr.Textbox(show_label=False, interactive=False, container=False)
                    c["videos_html"] = gr.HTML("")
                    c["marked_summary_tb"] = gr.Textbox(label="Excluded from the export", interactive=False)
                    c["ep_shown_state"] = gr.State(0)
                # ── Señales ──────────────────────────────────────────────────
                with gr.Tab("Signal plots", render_children=True):
                    gr.Markdown("Plot action / observation.state signals of the episode shown in "
                                "**Episodes**. Click a plot to move the videos to that instant.",
                                elem_classes=["v2-muted"])
                    with gr.Row(equal_height=True):
                        c["new_plot_vars"] = gr.Dropdown(choices=[], multiselect=True, label="Signals", scale=4)
                        c["new_plot_title"] = gr.Textbox(label="Title (optional)", scale=2)
                        c["confirm_plot_btn"] = gr.Button("➕ Add plot", variant="primary", scale=1)
                    c["dynamic_plots_state"] = gr.State([])
                    _plots(c)
                # ── Qué conservar ────────────────────────────────────────────
                with gr.Tab("Data to keep", render_children=True):
                    gr.Markdown("Unchecked items are removed from the exported copy. "
                                "The recorded dataset is never modified.", elem_classes=["v2-muted"])
                    with gr.Row(equal_height=True):
                        (c["whole_all_btn"], c["whole_none_btn"], c["whole_search_tb"],
                         c["whole_feature_cbg"]) = _signal_list("Features", "cbg-whole", "🔍 Search features")
                        (c["action_all_btn"], c["action_none_btn"], c["action_search_tb"],
                         c["dims_action_cbg"]) = _signal_list("action dimensions", "cbg-action",
                                                              "🔍 Search dimensions")
                        (c["state_all_btn"], c["state_none_btn"], c["state_search_tb"],
                         c["dims_state_cbg"]) = _signal_list("observation.state dimensions", "cbg-state",
                                                             "🔍 Search dimensions")
                # ── Instrucción ──────────────────────────────────────────────
                with gr.Tab("Instruction", render_children=True):
                    gr.Markdown("Language instruction stored with every episode.", elem_classes=["v2-muted"])
                    with gr.Row(elem_classes=["task-row"]):
                        c["task_tb"] = gr.Textbox(value="", show_label=False, interactive=False, scale=10)
                        c["task_edit_btn"] = gr.Button("✏️ Edit", min_width=90, scale=0, elem_classes=["task-btn"])
                        c["task_save_btn"] = gr.Button("💾 Save", min_width=90, scale=0, interactive=False,
                                                       elem_classes=["task-btn"])
                    c["task_dirty_state"] = gr.State(False)

    c["tab"] = tab
    return c


def _plots(c: dict) -> None:
    @gr.render(inputs=[c["dynamic_plots_state"], c["ep_shown_state"]])
    def _render(plots, ep_idx):
        if not plots:
            gr.Markdown("*No plots yet.*", elem_classes=["v2-muted"])
        for i, p in enumerate(plots):
            title = p["title"] or f"Plot {i + 1}"
            with gr.Column(elem_classes=["v2-card"]):
                with gr.Row():
                    gr.Markdown(f"**{title}** — {', '.join(p['vars'])}")
                    html_btn = gr.DownloadButton("⬇ HTML", size="sm", scale=0)
                    del_btn = gr.Button("✕ Remove", size="sm", scale=0)
                fig = build_custom_plotly_chart(int(ep_idx), p["vars"], title=title)
                if fig:
                    gr.Plot(value=fig)
                else:
                    gr.Markdown("*No data for these signals.*")

            def _export(ep=int(ep_idx), v=p["vars"], t=title):
                f = build_custom_plotly_chart(ep, v, title=t)
                if not f:
                    return None
                path = tempfile.mktemp(suffix=".html", prefix="plot_")
                f.write_html(path, include_plotlyjs="cdn")
                return path

            html_btn.click(_export, None, [html_btn])
            del_btn.click(lambda cur, k=i: cur[:k] + cur[k + 1:], [c["dynamic_plots_state"]],
                          [c["dynamic_plots_state"]])


def wire(c: dict, export: dict) -> None:
    """`export`: componentes de Export que load_viz_dataset también reinicia."""
    nav = [c["ep_num_nb"], c["ep_info_tb"], c["videos_html"], c["ep_delete_btn"], c["marked_summary_tb"],
           c["ep_shown_state"]]
    # Mismo orden que espera curation.load_viz_dataset (ver launcher_gui/app.py)
    cur = [c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"], c["task_tb"],
           c["task_edit_btn"], c["task_save_btn"], c["task_dirty_state"], export["new_repo_id_tb"],
           c["new_plot_vars"], c["dynamic_plots_state"], c["new_plot_title"],
           export["package_status"], export["curation_log_box"], export["push_status"],
           export["push_link_html"], export["hf_repo_id_tb"]]
    dd_status = [c["dataset_dd"], c["viz_status"]]
    c["load_outputs"] = [c["dataset_col"], c["viz_status"]] + nav + cur  # también lo usa Record

    c["tab"].select(curation.refresh_datasets, [c["viz_root_tb"], c["dataset_dd"]], dd_status)
    c["tab"].select(curation.close_delete_dialog, None, [c["delete_confirm_grp"], c["delete_target_state"]])
    c["refresh_btn"].click(curation.refresh_datasets, [c["viz_root_tb"], c["dataset_dd"]], dd_status)
    c["browse_dir_ev"].then(curation.refresh_datasets, [c["viz_root_tb"]], dd_status)
    c["last_btn"].click(curation.use_last_dataset, [c["viz_root_tb"]], [c["dataset_dd"]])

    c["delete_ds_btn"].click(curation.ask_delete_dataset, [c["dataset_dd"]],
                             [c["delete_confirm_grp"], c["delete_confirm_md"], c["delete_target_state"]])
    c["delete_cancel_btn"].click(curation.close_delete_dialog, None,
                                 [c["delete_confirm_grp"], c["delete_target_state"]])
    c["delete_yes_btn"].click(
        curation.delete_dataset, [c["delete_target_state"], c["viz_root_tb"]],
        [c["delete_confirm_grp"], c["dataset_dd"], c["viz_status"], c["dataset_col"], c["delete_target_state"]],
    ).then(lambda: gr.update(visible=state.viz_dataset_path is None or not state.viz_ep_rows),
           None, [c["empty_md"]])

    c["load_ds_btn"].click(
        curation.load_viz_dataset, [c["dataset_dd"]], [c["dataset_col"], c["viz_status"]] + nav + cur,
    ).then(
        lambda: ("", "", "", gr.update(visible=not state.viz_ep_rows)),
        None, [c["whole_search_tb"], c["action_search_tb"], c["state_search_tb"], c["empty_md"]],
    )

    c["ep_prev_btn"].click(curation.ep_prev, [c["ep_shown_state"]], nav)
    c["ep_next_btn"].click(curation.ep_next, [c["ep_shown_state"]], nav)
    c["ep_num_nb"].submit(curation.ep_goto, [c["ep_num_nb"], c["ep_shown_state"]], nav)
    c["ep_delete_btn"].click(curation.toggle_delete_episode, [c["ep_shown_state"]],
                             [c["ep_delete_btn"], c["marked_summary_tb"]])

    def _add_plot(plots, vars_, title):
        if not vars_:
            gr.Warning("Pick at least one signal.")
            return plots, gr.update(), gr.update()
        return plots + [{"vars": vars_, "title": (title or "").strip()}], [], ""

    c["confirm_plot_btn"].click(_add_plot, [c["dynamic_plots_state"], c["new_plot_vars"], c["new_plot_title"]],
                                [c["dynamic_plots_state"], c["new_plot_vars"], c["new_plot_title"]])

    c["whole_all_btn"].click(curation.select_all_whole, None, [c["whole_feature_cbg"]])
    c["whole_none_btn"].click(curation.select_none_list, None, [c["whole_feature_cbg"]])
    c["action_all_btn"].click(curation.select_all_action_dims, None, [c["dims_action_cbg"]])
    c["action_none_btn"].click(curation.select_none_list, None, [c["dims_action_cbg"]])
    c["state_all_btn"].click(curation.select_all_state_dims, None, [c["dims_state_cbg"]])
    c["state_none_btn"].click(curation.select_none_list, None, [c["dims_state_cbg"]])
    for tb, cbg_id in (("whole_search_tb", "cbg-whole"), ("action_search_tb", "cbg-action"),
                       ("state_search_tb", "cbg-state")):
        c[tb].change(fn=None, inputs=[c[tb]], js=_checkbox_filter_js(cbg_id))

    c["task_edit_btn"].click(curation.task_start_edit, [c["task_tb"]],
                             [c["task_tb"], c["task_edit_btn"], c["task_save_btn"]])
    c["task_save_btn"].click(curation.task_save, [c["task_tb"]],
                             [c["task_tb"], c["task_edit_btn"], c["task_save_btn"], c["task_dirty_state"]])
