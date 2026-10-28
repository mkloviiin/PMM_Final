"""Pestana 'Control de episodios': generica, no depende del robot activo. El total de
la barra de progreso lo fija session.poll_status (state.ep_total, que sigue al "Number of
episodes" de la plataforma activa); num_eps_component solo da el valor inicial.
"""

from __future__ import annotations

import gradio as gr

from .. import live_view

EPISODES_TAB_ID = "episodes"


def build_episodes_tab(num_eps_component: gr.components.Component):
    # Pestañas y secciones numeradas en el orden de uso (pedido del cuestionario SUS)
    with gr.Tab("2. Episode Control", id=EPISODES_TAB_ID) as episodes_tab:
        gr.Markdown("### 1. Status")
        with gr.Row():
            status_lbl = gr.Textbox(
                value="Idle", label="Status",
                interactive=False, scale=2,
            )
            progress_lbl = gr.Textbox(
                value="Episode: 0 / 0  (0%)", label="Progress",
                interactive=False, scale=3,
            )
        progress_bar = gr.Slider(
            0, num_eps_component.value, value=0, label="Completed", interactive=False,
            buttons=[], elem_classes=["progress-completado"],
        )

        gr.Markdown("---")
        gr.Markdown("### 2. Episode controls")
        with gr.Row():
            start_btn = gr.Button("RECORD episode", variant="primary", scale=2, elem_classes=["btn-green"])
            stop_btn = gr.Button("STOP episode", variant="secondary", scale=2, elem_classes=["btn-red"])
            exit_btn = gr.Button("End session", variant="stop", scale=1)
        cmd_feedback = gr.Textbox(label="Response", interactive=False)

        gr.Markdown("### 3. Live view")
        # With VR: what the headset shows + its front screen; without VR: robot eye + scene camera
        gr.HTML(live_view.html())

        gr.Markdown("### 4. Logbook")
        with gr.Row(equal_height=True):
            note_tb = gr.Textbox(
                label="Note", lines=2, scale=5,
                placeholder="Observation about the current episode (saved with its episode number)",
            )
            add_note_btn = gr.Button("Add note", scale=1)
        notes_md = gr.Markdown("*No notes yet.*")

        gr.Markdown("---")
        gr.Markdown("### 5. Real-time metrics")
        with gr.Row():
            fps_nb = gr.Number(label="FPS", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=1, scale=1)
            lat_nb = gr.Number(label="Latency (ms)", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=1, scale=1)
            size_nb = gr.Number(label="Dataset size (MB)", elem_classes=["metric-nb"], value=0.0, interactive=False, precision=2, scale=1)

        with gr.Accordion("Per-episode metrics (saved in <dataset>/hub_session/)", open=True):
            episode_metrics_plot = gr.Plot(label="Per-episode metrics", show_label=False)

        gr.Markdown("---")
        gr.Markdown("### 6. Live log")
        log_box = gr.Textbox(
            label="Live log  (last 80 lines)",
            lines=22,
            max_lines=22,
            interactive=False,
        )

    return dict(
        status_lbl=status_lbl, progress_lbl=progress_lbl, progress_bar=progress_bar,
        start_btn=start_btn, stop_btn=stop_btn, exit_btn=exit_btn,
        cmd_feedback=cmd_feedback, log_box=log_box,
        fps_nb=fps_nb, lat_nb=lat_nb, size_nb=size_nb,
        episode_metrics_plot=episode_metrics_plot, note_tb=note_tb, add_note_btn=add_note_btn,
        notes_md=notes_md, tab=episodes_tab,
    )
