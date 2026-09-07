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
            fps_nb = gr.Number(label="FPS", value=0.0, interactive=False, precision=1, scale=1, show_label=True, container=True, elem_classes=["no-spinner"])
            lat_nb = gr.Number(label="Latency (ms)", value=0.0, interactive=False, precision=1, scale=1, elem_classes=["no-spinner"])
            size_nb = gr.Number(label="Dataset size (MB)", value=0.0, interactive=False, precision=2, scale=1, elem_classes=["no-spinner"])

        gr.Markdown("---")
        with gr.Accordion("❓ Help: controls", open=False):
            gr.Markdown(
                "**Simulation control:**\n"
                "- `1`: Open right hand\n"
                "- `2`: Close right hand\n"
                "- `3`: Stop right hand\n"
                "- `4`: Open left hand\n"
                "- `5`: Close left hand\n"
                "- `6`: Stop left hand\n"
                "- `R`: Randomize scene object positions\n"
                "- `Space`: Reset mocap targets to XML pose\n"
                "- `Ctrl` + right-click + drag: move the target (hand position)\n"
                "- `Ctrl` + left-click + drag: rotate the target (hand angle)\n\n"
                "**VR control (Meta Quest):**\n"
                "- `A`: Start recording episode\n"
                "- `B` (short tap): Stop and save episode\n"
                "- `B` (hold 1.5s+): Discard episode and re-record\n"
                "- Right index trigger: Close right hand\n"
                "- Right grip: Open right hand\n"
                "- Left index trigger: Close left hand\n"
                "- Left grip: Open left hand\n"
                "- `Y`: Randomize scene object positions"
            )
        gr.Markdown("### Episode Controls")
        with gr.Row():
            start_btn = gr.Button("RECORD episode", variant="primary", scale=2)
            stop_btn = gr.Button("STOP episode", variant="secondary", scale=2)
            exit_btn = gr.Button("End session", variant="stop", scale=1)
        cmd_feedback = gr.Textbox(label="Response", interactive=False)
        with gr.Row():
            episode_note_input_tb = gr.Textbox(label="Episode note", placeholder="Optional notes for this episode", scale=4)
            copy_note_recording_btn = gr.Button("Copy previous", scale=1)
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
        episode_note_input_tb=episode_note_input_tb, copy_note_recording_btn=copy_note_recording_btn,
        fps_nb=fps_nb, lat_nb=lat_nb, size_nb=size_nb
    )
