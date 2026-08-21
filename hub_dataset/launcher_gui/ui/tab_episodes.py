"""Pestana 'Control de episodios': generica, no depende del robot activo mas
alla del componente de "numero de episodios" que le pasa cada robot para
acotar la barra de progreso.
"""

from __future__ import annotations

import gradio as gr


def build_episodes_tab(num_eps_component: gr.components.Component):
    with gr.Tab("Episode Control"):
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

        num_eps_component.change(
            fn=lambda n: gr.update(maximum=max(int(n or 1), 1)),
            inputs=[num_eps_component],
            outputs=[progress_bar],
        )

        gr.Markdown("---")
        gr.Markdown("### Real-time Metrics")
        with gr.Row():
            fps_nb = gr.Number(label="FPS", value=0.0, interactive=False, precision=1, scale=1)
            lat_nb = gr.Number(label="Latency (ms)", value=0.0, interactive=False, precision=1, scale=1)
            size_nb = gr.Number(label="Dataset size (MB)", value=0.0, interactive=False, precision=2, scale=1)

        gr.Markdown("---")
        gr.Markdown("### Episode Controls")
        with gr.Row():
            start_btn = gr.Button("RECORD episode", variant="primary", scale=2)
            stop_btn = gr.Button("STOP episode", variant="secondary", scale=2)
            exit_btn = gr.Button("End session", variant="stop", scale=1)
        cmd_feedback = gr.Textbox(label="Response", interactive=False)

        gr.Markdown("---")
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
        fps_nb=fps_nb, lat_nb=lat_nb, size_nb=size_nb
    )
