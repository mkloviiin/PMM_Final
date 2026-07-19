"""Pestana 'Visualizar y curar': genera todos los componentes y devuelve un
dict con los handles que necesita app.py para conectar los eventos definidos
en launcher_gui/curation.py y launcher_gui/hub_push.py.
"""

from __future__ import annotations

import gradio as gr

from ..paths import HUB_ROOT
from ..curation import build_custom_plotly_chart


def build_curation_tab():
    with gr.Tab("Visualizar y curar"):
        gr.Markdown("Carga un dataset grabado, explora sus episodios y elige que se conserva en la version final.")

        with gr.Row():
            viz_root_tb = gr.Textbox(
                value=str(HUB_ROOT.parent / "data"),
                label="Directorio de datasets",
                scale=4,
            )
            refresh_btn = gr.Button("Actualizar lista", scale=1)

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=[], value=None,
                label="Dataset",
                info="Selecciona un dataset de la lista o escribe la ruta directamente",
                allow_custom_value=True,
                scale=4,
            )
            last_btn = gr.Button("Usar ultimo grabado", scale=1)

        with gr.Row():
            load_ds_btn = gr.Button("Cargar dataset", variant="primary", scale=1)
            viz_status = gr.Textbox(label="Estado", interactive=False, scale=3)

        gr.Markdown("---")

        with gr.Row():
            ep_prev_btn = gr.Button("◀", scale=0, min_width=50)
            gr.Markdown("**Episodio**")
            ep_num_nb = gr.Number(value=0, show_label=False, precision=0, scale=0, min_width=80)
            ep_next_btn = gr.Button("▶", scale=0, min_width=50)
            ep_delete_btn = gr.Button("🗑️ Eliminar este episodio", scale=2)
        ep_info_tb = gr.Textbox(label="Info del episodio", interactive=False)
        marked_summary_tb = gr.Textbox(label="Episodios excluidos del dataset final", interactive=False)

        with gr.Row():
            vid0 = gr.Video(label="Camara 0", autoplay=True, loop=True)
            vid1 = gr.Video(label="Camara 1", autoplay=True, loop=True)
            vid2 = gr.Video(label="Camara 2", autoplay=True, loop=True)

        with gr.Row():
            gr.Markdown("### Gráficos Dinámicos")
            add_plot_btn = gr.Button("➕ Agregar Gráfico", size="sm", scale=0)
            
        with gr.Group(visible=False) as new_plot_group:
            gr.Markdown("Selecciona las variables para el nuevo gráfico:")
            new_plot_vars = gr.Dropdown(choices=[], multiselect=True, label="Variables")
            with gr.Row():
                confirm_plot_btn = gr.Button("Generar Gráfico", variant="primary")
                cancel_plot_btn = gr.Button("Cancelar")
                
        dynamic_plots_state = gr.State([])

        @gr.render(inputs=[dynamic_plots_state, ep_num_nb])
        def render_dynamic_plots(plots_config, ep_idx):
            for i, vars_list in enumerate(plots_config):
                with gr.Group():
                    with gr.Row():
                        gr.Markdown(f"**Gráfico Personalizado {i+1}**: {', '.join(vars_list)}")
                        del_btn = gr.Button("❌ Eliminar", size="sm", scale=0)
                    
                    fig = build_custom_plotly_chart(int(ep_idx), vars_list)
                    if fig:
                        gr.Plot(value=fig)
                    else:
                        gr.Markdown("*(No hay datos o variables no disponibles)*")

                    def make_del_fn(index):
                        def del_fn(current_state):
                            new_state = current_state.copy()
                            new_state.pop(index)
                            return new_state
                        return del_fn
                    
                    del_btn.click(
                        fn=make_del_fn(i),
                        inputs=[dynamic_plots_state],
                        outputs=[dynamic_plots_state]
                    )

        gr.Markdown("---")
        gr.Markdown("### Senales a integrar en el dataset final")

        with gr.Group():
            with gr.Row():
                gr.Markdown("**Features completas** (camaras, action, observation.state)")
                whole_all_btn = gr.Button("Todo", size="sm", scale=0)
                whole_none_btn = gr.Button("Ninguno", size="sm", scale=0)
            whole_feature_cbg = gr.CheckboxGroup(choices=[], value=[], show_label=False)

        with gr.Group():
            with gr.Row():
                gr.Markdown("**action** — dimensiones")
                action_all_btn = gr.Button("Todo", size="sm", scale=0)
                action_none_btn = gr.Button("Ninguno", size="sm", scale=0)
            dims_action_cbg = gr.CheckboxGroup(choices=[], value=[], show_label=False)

        with gr.Group():
            with gr.Row():
                gr.Markdown("**observation.state** — dimensiones")
                state_all_btn = gr.Button("Todo", size="sm", scale=0)
                state_none_btn = gr.Button("Ninguno", size="sm", scale=0)
            dims_state_cbg = gr.CheckboxGroup(choices=[], value=[], show_label=False)

        gr.Markdown("---")
        with gr.Row():
            gr.Markdown("### Instruccion (Language Instruction) — global para todo el dataset")
            task_edit_btn = gr.Button("✏️", scale=0, min_width=50)
            task_save_btn = gr.Button("💾 Guardar", scale=0, visible=False)
        task_tb = gr.Textbox(value="", show_label=False, interactive=False)
        task_dirty_state = gr.State(False)

        gr.Markdown("---")
        gr.Markdown("### Empaquetado")
        with gr.Row():
            new_repo_id_tb = gr.Textbox(value="", label="Repo ID final (local)")
            curated_root_tb = gr.Textbox(value=str(HUB_ROOT.parent / "data_curated"), label="Carpeta raiz para el dataset curado")
        package_btn = gr.Button("Empaquetar dataset", variant="primary", size="lg")
        package_status = gr.Textbox(label="Estado del empaquetado", interactive=False)
        curation_log_box = gr.Textbox(label="Log de curacion", lines=10, max_lines=10, interactive=False)

        gr.Markdown("---")
        gr.Markdown(
            "### Subir a Hugging Face Hub\n"
            "Primero tenes que empaquetar un dataset (seccion de arriba) antes de poder subirlo."
        )
        with gr.Accordion("❓ Como consigo mi token de Hugging Face", open=False):
            gr.Markdown(
                "1. Anda a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
                "y crea un token nuevo con permiso **Write** (o, si usas un token *fine-grained*, "
                "activa el permiso de escritura sobre repos de datasets).\n"
                "2. Copia el token (empieza con `hf_...`) y pegalo en el campo de abajo. "
                "No se guarda en disco, solo se usa para esta subida.\n"
                "3. El **Repo ID** debe empezar con tu usuario real de Hugging Face — revisalo en "
                "[huggingface.co/settings/profile](https://huggingface.co/settings/profile), campo "
                "\"Username\" — o una organizacion a la que pertenezcas con permiso de escritura. "
                "No uses tu usuario del sistema operativo ni `local/...`: si no coincide con tu cuenta "
                "de Hugging Face vas a ver un error 403 Forbidden."
            )
        with gr.Row():
            hf_repo_id_tb = gr.Textbox(
                label="Repo ID en Hugging Face",
                placeholder="tu_usuario/nombre_dataset",
                scale=3,
            )
            hf_token_tb = gr.Textbox(
                label="Token de Hugging Face (sesion, no se guarda en disco)",
                type="password", placeholder="hf_...",
                scale=3,
            )
            hf_private_cb = gr.Checkbox(value=True, label="Privado", scale=1)
        push_btn = gr.Button("Subir a Hugging Face", variant="stop", size="lg")
        push_status = gr.Textbox(label="Estado de la subida", interactive=False)

    return dict(
        viz_root_tb=viz_root_tb, refresh_btn=refresh_btn,
        dataset_dd=dataset_dd, last_btn=last_btn,
        load_ds_btn=load_ds_btn, viz_status=viz_status,
        ep_prev_btn=ep_prev_btn, ep_num_nb=ep_num_nb, ep_next_btn=ep_next_btn,
        ep_delete_btn=ep_delete_btn, ep_info_tb=ep_info_tb, marked_summary_tb=marked_summary_tb,
        vid0=vid0, vid1=vid1, vid2=vid2,
        add_plot_btn=add_plot_btn, new_plot_group=new_plot_group, new_plot_vars=new_plot_vars,
        confirm_plot_btn=confirm_plot_btn, cancel_plot_btn=cancel_plot_btn,
        dynamic_plots_state=dynamic_plots_state,
        whole_all_btn=whole_all_btn, whole_none_btn=whole_none_btn, whole_feature_cbg=whole_feature_cbg,
        action_all_btn=action_all_btn, action_none_btn=action_none_btn, dims_action_cbg=dims_action_cbg,
        state_all_btn=state_all_btn, state_none_btn=state_none_btn, dims_state_cbg=dims_state_cbg,
        task_edit_btn=task_edit_btn, task_save_btn=task_save_btn,
        task_tb=task_tb, task_dirty_state=task_dirty_state,
        new_repo_id_tb=new_repo_id_tb, curated_root_tb=curated_root_tb,
        package_btn=package_btn, package_status=package_status, curation_log_box=curation_log_box,
        hf_repo_id_tb=hf_repo_id_tb, hf_token_tb=hf_token_tb, hf_private_cb=hf_private_cb,
        push_btn=push_btn, push_status=push_status,
    )
