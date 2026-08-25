"""Construccion de la UI del Hub: ensambla la pantalla de bienvenida y las
pestanas, y conecta cada componente con session.py / curation.py / hub_push.py
y con el `launch()` de cada robot disponible en robots.registry.

Agregar un robot nuevo: basta con sumarlo al
registro (robots/registry.py) con su propio `build_config_form`/`launch`/
`config_input_order` (ver robots/base.py). Este modulo itera ese registro
para construir la pantalla de bienvenida, las columnas de "Configuración" y
el wiring del boton de lanzamiento, sin conocer nada especifico de ningun
robot en particular.
"""

from __future__ import annotations

import gradio as gr

from . import curation, hub_push, session
from .robots.registry import REGISTRY
from .ui.tab_config import build_config_tab
from .ui.tab_curation import build_curation_tab
from .ui.tab_episodes import build_episodes_tab
from .ui.welcome import build_welcome_screen

WELCOME_CSS = """
.welcome-screen {
    justify-content: center !important;
    align-items: center !important;
    min-height: 85vh !important;
}
.welcome-institutional-logos {
    justify-content: center !important;
    align-items: center !important;
    gap: 2.5rem !important;
    margin-top: -5rem !important;
}
.toast-wrap {
    top: 50% !important;
    bottom: auto !important;
    left: 50% !important;
    right: auto !important;
    transform: translate(-50%, -50%) !important;
    align-items: center !important;
}
.progress-completado input[data-testid="number-input"] {
    display: none !important;
}
/* Scrollable vertical CheckboxGroup */
.scrollable-list .wrap {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    max-height: 200px !important;
    overflow-y: auto !important;
    gap: 2px !important;
    padding: 4px 8px !important;
}
.scrollable-list .wrap label {
    width: 100% !important;
    min-width: unset !important;
    padding: 3px 0 !important;
}
/* Episode navigation row: force all children to the same height */
.ep-nav-row > div {
    align-items: stretch !important;
}
.ep-nav-row button,
.ep-nav-row input[type="number"],
.ep-nav-row .wrap {
    height: 42px !important;
    min-height: 42px !important;
    box-sizing: border-box !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="iCub MuJoCo Launcher") as demo:
        welcome_col, welcome_buttons = build_welcome_screen()



        with gr.Column(visible=False) as main_col:
            back_btn = gr.Button("⬅ Back to robot selection", size="sm")
            gr.Markdown("# iCub MuJoCo Simulation  —  Experimentation Hub")

            with gr.Tabs():
                config_columns, config_forms = build_config_tab()

                available_robots = [r for r in REGISTRY if r.available and r.build_config_form]
                episodes = build_episodes_tab(config_forms[available_robots[0].id]["num_eps"])
                c = build_curation_tab()

        # ── Navegacion bienvenida <-> configuracion ───────────────────────
        back_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[welcome_col, main_col],
        )
        for robot in REGISTRY:
            btn = welcome_buttons[robot.id]
            if robot.id in config_columns:
                def _select_robot(robot_id=robot.id):
                    return [
                        gr.update(visible=False), gr.update(visible=True),
                        *(gr.update(visible=(rid == robot_id)) for rid in config_columns),
                    ]
                btn.click(fn=_select_robot, outputs=[welcome_col, main_col, *config_columns.values()])
            else:
                label = robot.label
                btn.click(fn=lambda label=label: gr.Warning(f"{label} — not yet available."))

        # ── Lanzamiento de sesion (uno por robot disponible) ──────────────
        for robot in available_robots:
            form = config_forms[robot.id]
            inputs = [form[key] for key in robot.config_input_order]
            form["launch_btn"].click(fn=robot.launch, inputs=inputs, outputs=[form["launch_status"]])

        # ── Control de episodios (generico, no depende del robot) ────────
        episodes["start_btn"].click(fn=session.cmd_start, outputs=[episodes["cmd_feedback"]])
        episodes["stop_btn"].click(fn=session.cmd_stop, outputs=[episodes["cmd_feedback"]])
        episodes["exit_btn"].click(fn=session.cmd_exit, outputs=[episodes["cmd_feedback"]])

        gr.Timer(0.5).tick(
            fn=session.poll_status,
            outputs=[episodes["status_lbl"], episodes["progress_lbl"],
                     episodes["progress_bar"], episodes["log_box"],
                     episodes["fps_nb"], episodes["lat_nb"], episodes["size_nb"],
                     back_btn],
        )

        # ── Visualizar y curar (generico, no depende del robot) ──────────
        nav_outputs = [c["ep_num_nb"], c["ep_info_tb"], c["vid0"], c["vid1"], c["vid2"],
                       c["ep_delete_btn"], c["marked_summary_tb"]]
        curation_outputs = [c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"],
                            c["task_tb"], c["task_edit_btn"], c["task_save_btn"],
                            c["task_dirty_state"], c["new_repo_id_tb"],
                            c["new_plot_vars"], c["dynamic_plots_state"], c["new_plot_title"]]

        c["refresh_btn"].click(fn=curation.refresh_datasets, inputs=[c["viz_root_tb"]],
                                outputs=[c["dataset_dd"], c["viz_status"]])
        c["last_btn"].click(fn=curation.use_last_dataset, inputs=[c["viz_root_tb"]], outputs=[c["dataset_dd"]])
        c["load_ds_btn"].click(
            fn=curation.load_viz_dataset, inputs=[c["dataset_dd"]],
            outputs=[c["viz_status"]] + nav_outputs + curation_outputs,
        )

        c["add_plot_btn"].click(
            fn=lambda: gr.update(visible=True),
            inputs=[], outputs=[c["new_plot_group"]]
        )
        c["cancel_plot_btn"].click(
            fn=lambda: (gr.update(visible=False), gr.update(value=[]), gr.update(value="")),
            inputs=[], outputs=[c["new_plot_group"], c["new_plot_vars"], c["new_plot_title"]]
        )
        
        def confirm_plot(plots_state, selected_vars, title):
            if not selected_vars:
                return plots_state, gr.update(visible=False), gr.update(value=[]), gr.update(value="")
            new_state = plots_state.copy()
            new_state.append({"vars": selected_vars, "title": title.strip()})
            return new_state, gr.update(visible=False), gr.update(value=[]), gr.update(value="")

        c["confirm_plot_btn"].click(
            fn=confirm_plot,
            inputs=[c["dynamic_plots_state"], c["new_plot_vars"], c["new_plot_title"]],
            outputs=[c["dynamic_plots_state"], c["new_plot_group"], c["new_plot_vars"], c["new_plot_title"]]
        )



        c["ep_prev_btn"].click(fn=curation.ep_prev, inputs=[c["ep_num_nb"]], outputs=nav_outputs)
        c["ep_next_btn"].click(fn=curation.ep_next, inputs=[c["ep_num_nb"]], outputs=nav_outputs)
        c["ep_num_nb"].submit(fn=curation.ep_goto, inputs=[c["ep_num_nb"]], outputs=nav_outputs)
        c["ep_delete_btn"].click(fn=curation.toggle_delete_episode, inputs=[c["ep_num_nb"]],
                                  outputs=[c["ep_delete_btn"], c["marked_summary_tb"]])

        c["whole_all_btn"].click(fn=curation.select_all_whole, outputs=[c["whole_feature_cbg"]])
        c["whole_none_btn"].click(fn=curation.select_none_list, outputs=[c["whole_feature_cbg"]])
        c["action_all_btn"].click(fn=curation.select_all_action_dims, outputs=[c["dims_action_cbg"]])
        c["action_none_btn"].click(fn=curation.select_none_list, outputs=[c["dims_action_cbg"]])
        c["state_all_btn"].click(fn=curation.select_all_state_dims, outputs=[c["dims_state_cbg"]])
        c["state_none_btn"].click(fn=curation.select_none_list, outputs=[c["dims_state_cbg"]])

        c["whole_search_tb"].change(
            fn=curation.filter_whole_features,
            inputs=[c["whole_search_tb"], c["whole_feature_cbg"]],
            outputs=[c["whole_feature_cbg"]],
        )
        c["action_search_tb"].change(
            fn=curation.filter_action_dims,
            inputs=[c["action_search_tb"], c["dims_action_cbg"]],
            outputs=[c["dims_action_cbg"]],
        )
        c["state_search_tb"].change(
            fn=curation.filter_state_dims,
            inputs=[c["state_search_tb"], c["dims_state_cbg"]],
            outputs=[c["dims_state_cbg"]],
        )

        c["task_edit_btn"].click(fn=curation.task_start_edit,
                                  outputs=[c["task_tb"], c["task_edit_btn"], c["task_save_btn"]])
        c["task_save_btn"].click(
            fn=curation.task_save, inputs=[c["task_tb"]],
            outputs=[c["task_tb"], c["task_edit_btn"], c["task_save_btn"], c["task_dirty_state"]],
        )

        c["package_btn"].click(
            fn=curation.run_packaging,
            inputs=[c["dataset_dd"], c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"],
                    c["task_tb"], c["task_dirty_state"], c["new_repo_id_tb"], c["curated_root_tb"]],
            outputs=[c["package_status"]],
        )
        gr.Timer(0.5).tick(fn=curation.poll_curation_status, outputs=[c["package_status"], c["curation_log_box"]])

        c["push_btn"].click(
            fn=hub_push.push_to_hub_action,
            inputs=[c["hf_repo_id_tb"], c["hf_token_tb"], c["hf_private_cb"]],
            outputs=[c["push_status"]],
        )
        gr.Timer(0.5).tick(fn=hub_push.poll_push_status, outputs=[c["push_status"]])

    return demo
