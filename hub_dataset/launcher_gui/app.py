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
from .ui.tab_config import CONFIG_TAB_ID, ROBOT_HIDDEN, ROBOT_SHOWN, build_config_tab
from .ui.tab_curation import build_curation_tab
from .ui.tab_episodes import build_episodes_tab
from .ui.welcome import build_welcome_screen
from .ui.input_filters import INPUT_FILTER_JS
from .video_sync import PLAYER_CSS, SYNC_JS

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
/* Campos con botón de carpeta (ui/dir_picker.py): el botón va en `buttons=` del Textbox
   (barra superior); aquí se baja dentro del cuadro de texto, como la flecha de un Dropdown. */
.dir-tb textarea {
    padding-right: 48px !important;
}
.dir-tb .icon-button-wrapper {
    position: absolute !important;
    top: auto !important;
    bottom: 11px !important;
    right: 22px !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 5 !important;
}
.dir-tb .custom-button {
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    background: transparent url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%23f97316' d='M10 4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z'/%3E%3C/svg%3E") center / 22px 22px no-repeat !important;
}
.dir-tb .custom-button:hover {
    background-color: rgba(127, 127, 127, 0.18) !important;
}
/* Configuration: con "Enable VR" marcado (aparecen cable + IP) las dos columnas quedan de la
   misma altura; sin VR cada columna mide lo suyo (equal_height fijo estiraba el panel derecho). */
.cfg-row:has(.vr-cb input:checked) {
    align-items: stretch !important;
}
/* Métricas en vivo (solo lectura): sin las flechitas del campo numérico (se veían como un cuadro gris) */
.metric-nb input[type=number]::-webkit-inner-spin-button,
.metric-nb input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
.metric-nb input[type=number] {
    -moz-appearance: textfield !important;
    appearance: textfield !important;
}
.metric-nb button {
    display: none !important;
}
/* Pop-up de confirmación (borrar dataset): centrado sobre la pantalla; el box-shadow gigante
   hace de fondo oscurecido. */
.pmm-modal {
    position: fixed !important;
    inset: 0 !important;
    margin: auto !important;
    width: min(520px, 90vw) !important;
    height: fit-content !important;
    max-height: 90vh !important;
    padding: 24px 28px 20px !important;
    gap: 14px !important;
    z-index: 1000 !important;
    border-radius: 14px !important;
    background: var(--block-background-fill) !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45), 0 0 0 100vmax rgba(0, 0, 0, 0.55) !important;
}
.pmm-modal h3 {
    margin-top: 0 !important;
}
.pmm-modal code {
    font-size: 0.95rem !important;
    word-break: break-all !important;
}
/* Botones de acción con color fijo (independiente del tema): verde = cargar/grabar, rojo = borrar/detener */
button.btn-green, .btn-green button {
    background: #2f9e44 !important;
    border-color: #2f9e44 !important;
    color: #fff !important;
}
button.btn-green:hover, .btn-green button:hover {
    background: #2b8a3e !important;
    border-color: #2b8a3e !important;
}
button.btn-red, .btn-red button {
    background: #e03131 !important;
    border-color: #e03131 !important;
    color: #fff !important;
}
button.btn-red:hover, .btn-red button:hover {
    background: #c92a2a !important;
    border-color: #c92a2a !important;
}
/* Configuration: formulario del robot no elegido (ver ui/tab_config.py) */
.robot-hidden {
    display: none !important;
}
/* Task instruction row: align items and fix button height to match textbox */
.task-row {
    align-items: center !important;
}
.task-row > div {
    align-items: center !important;
}
.task-btn button {
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    box-sizing: border-box !important;
}
"""
WELCOME_CSS += PLAYER_CSS


def _checkbox_filter_js(elem_id: str) -> str:
    """JS que oculta las casillas del CheckboxGroup `elem_id` que no contienen el texto buscado."""
    return """(q) => {
        const box = document.getElementById("%s");
        if (!box) return [];
        const t = (q || "").trim().toLowerCase();
        box.querySelectorAll("label").forEach(l => {
            if (!l.querySelector("input[type=checkbox]")) return;
            l.style.display = (!t || l.textContent.toLowerCase().includes(t)) ? "" : "none";
        });
        return [];
    }""" % elem_id


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="iCub Experimentation Hub", css=WELCOME_CSS) as demo:
        welcome_col, welcome_buttons = build_welcome_screen()
        # Sincroniza los videos de "Visualize & Curate" y dibuja el cursor de
        # tiempo sobre los graficos (ver video_sync.py).
        demo.load(fn=None, js=SYNC_JS)
        # Campos numéricos: solo dígitos (IP del VR: dígitos y puntos)
        demo.load(fn=None, js=INPUT_FILTER_JS)



        with gr.Column(visible=False) as main_col:
            back_btn = gr.Button("⬅ Back to robot selection", size="sm")
            gr.Markdown("# iCub  —  Experimentation Hub")

            with gr.Tabs() as main_tabs:
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
                # La columna del robot se muestra/oculta con la clase .robot-hidden, no con
                # visible= (ver ui/tab_config.py: con visible Gradio dejaba Configuration en blanco).
                # Al elegir robot se abre siempre la pestaña Configuration (aunque antes se haya
                # salido desde Visualize & Curate o Episode Control).
                def _select_robot(robot_id=robot.id):
                    return [
                        gr.update(visible=False), gr.update(visible=True),
                        gr.update(selected=CONFIG_TAB_ID),
                        *(gr.update(elem_classes=ROBOT_SHOWN if rid == robot_id else ROBOT_HIDDEN)
                          for rid in config_columns),
                    ]
                btn.click(fn=_select_robot,
                          outputs=[welcome_col, main_col, main_tabs, *config_columns.values()])
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
        episodes["tab"].select(fn=session.episode_panels,
                               outputs=[episodes["episode_metrics_plot"], episodes["notes_md"]])
        episodes["add_note_btn"].click(fn=session.add_note, inputs=[episodes["note_tb"]],
                                       outputs=[episodes["note_tb"], episodes["notes_md"]])

        gr.Timer(0.5).tick(
            fn=session.poll_status,
            outputs=[episodes["status_lbl"], episodes["progress_lbl"],
                     episodes["progress_bar"], episodes["log_box"],
                     episodes["fps_nb"], episodes["lat_nb"], episodes["size_nb"],
                     back_btn, episodes["episode_metrics_plot"], episodes["notes_md"]],
        )

        # ── Visualizar y curar (generico, no depende del robot) ──────────
        nav_outputs = [c["ep_num_nb"], c["ep_info_tb"], c["videos_html"],
                       c["ep_delete_btn"], c["marked_summary_tb"]]
        curation_outputs = [c["whole_feature_cbg"], c["dims_action_cbg"], c["dims_state_cbg"],
                            c["task_tb"], c["task_edit_btn"], c["task_save_btn"],
                            c["task_dirty_state"], c["new_repo_id_tb"],
                            c["new_plot_vars"], c["dynamic_plots_state"], c["new_plot_title"],
                            # Packaging & upload reset al cargar nuevo dataset
                            c["package_status"], c["curation_log_box"],
                            c["push_status"], c["push_link_html"],
                            c["hf_repo_id_tb"]]

        # Lista de datasets: automática al abrir la pestaña (y cada vez que se vuelve a ella),
        # conservando la selección; el botón hace lo mismo a pedido.
        c["tab"].select(fn=curation.refresh_datasets, inputs=[c["viz_root_tb"], c["dataset_dd"]],
                        outputs=[c["dataset_dd"], c["viz_status"]])
        c["refresh_btn"].click(fn=curation.refresh_datasets, inputs=[c["viz_root_tb"], c["dataset_dd"]],
                                outputs=[c["dataset_dd"], c["viz_status"]])
        # Borrar dataset: confirmación -> papelera
        c["delete_ds_btn"].click(fn=curation.ask_delete_dataset, inputs=[c["dataset_dd"]],
                                 outputs=[c["delete_confirm_grp"], c["delete_confirm_md"]])
        c["delete_cancel_btn"].click(fn=lambda: gr.update(visible=False), outputs=[c["delete_confirm_grp"]])
        c["delete_yes_btn"].click(fn=curation.delete_dataset, inputs=[c["dataset_dd"], c["viz_root_tb"]],
                                  outputs=[c["delete_confirm_grp"], c["dataset_dd"], c["viz_status"],
                                           c["dataset_col"]])
        c["last_btn"].click(fn=curation.use_last_dataset, inputs=[c["viz_root_tb"]], outputs=[c["dataset_dd"]])
        # Al elegir carpeta con el botón, refrescar la lista de datasets
        c["browse_dir_ev"].then(fn=curation.refresh_datasets, inputs=[c["viz_root_tb"]],
                                outputs=[c["dataset_dd"], c["viz_status"]])
        c["load_ds_btn"].click(
            fn=curation.load_viz_dataset, inputs=[c["dataset_dd"]],
            outputs=[c["dataset_col"], c["viz_status"]] + nav_outputs + curation_outputs,
        ).then(  # buscadores vacíos: la lista nueva se muestra completa
            fn=lambda: ("", "", ""),
            outputs=[c["whole_search_tb"], c["action_search_tb"], c["state_search_tb"]],
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

        # Buscadores de features/dimensiones: filtran en el navegador (ocultan las casillas que
        # no calzan) sin tocar las opciones, así la selección se conserva. Antes se filtraban
        # las opciones en Python dejando siempre las marcadas, y como al cargar un dataset
        # están todas marcadas, el buscador no ocultaba nada.
        for tb_key, cbg_id in (("whole_search_tb", "cbg-whole"), ("action_search_tb", "cbg-action"),
                               ("state_search_tb", "cbg-state")):
            c[tb_key].change(fn=None, inputs=[c[tb_key]], js=_checkbox_filter_js(cbg_id))

        c["task_edit_btn"].click(fn=curation.task_start_edit,
                                  inputs=[c["task_tb"]],
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
        gr.Timer(0.5).tick(fn=hub_push.poll_push_status, outputs=[c["push_status"], c["push_link_html"]])

    return demo
