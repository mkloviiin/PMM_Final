"""Pestana 'Visualizar y curar': genera todos los componentes y devuelve un
dict con los handles que necesita app.py para conectar los eventos definidos
en launcher_gui/curation.py y launcher_gui/hub_push.py.
"""

from __future__ import annotations

import gradio as gr

from ..paths import HUB_ROOT
from ..curation import build_custom_plotly_chart


def build_curation_tab():
    with gr.Tab("Visualize & Curate"):
        gr.Markdown("Load a recorded dataset, explore its episodes and choose what to keep in the final version.")

        with gr.Row():
            with gr.Column(scale=4):
                viz_root_tb = gr.Textbox(
                    value=str(HUB_ROOT.parent / "data"),
                    label="Datasets directory",
                )
                dataset_dd = gr.Dropdown(
                    choices=[], value=None,
                    label="Dataset",
                    info="Select a dataset from the list or type the path directly",
                    allow_custom_value=True,
                )
                viz_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                refresh_btn = gr.Button("Refresh list")
                last_btn = gr.Button("Use last recorded")
                load_ds_btn = gr.Button("Load dataset", variant="primary")

        gr.Markdown("---")

        with gr.Row(elem_classes=["ep-nav-row"]):
            ep_prev_btn = gr.Button("◄", scale=0, min_width=80)
            gr.HTML(
                "<div style='"
                "display:flex;align-items:center;justify-content:center;"
                "font-weight:bold;font-size:0.95rem;white-space:nowrap;"
                "padding:0 8px;'"
                ">Episode</div>"
            )
            ep_num_nb = gr.Number(
                value=0, show_label=False, precision=0,
                scale=0, min_width=80, container=False,
                minimum=0,
            )
            ep_next_btn = gr.Button("►", scale=0, min_width=80)
            ep_delete_btn = gr.Button("🗑️ Delete this episode", scale=2)
        ep_info_tb = gr.Textbox(label="Episode info", interactive=False)
        marked_summary_tb = gr.Textbox(label="Episodes excluded from the final dataset", interactive=False)

        with gr.Row():
            vid0 = gr.Video(label="Camera 0", autoplay=True, loop=True)
            vid1 = gr.Video(label="Camera 1", autoplay=True, loop=True)
            vid2 = gr.Video(label="Camera 2", autoplay=True, loop=True)

        with gr.Row():
            gr.Markdown("### Dynamic Plots")
            add_plot_btn = gr.Button("➕ Add Plot", size="sm", scale=0)
            
        with gr.Group(visible=False) as new_plot_group:
            gr.Markdown("Select variables and a title for the new plot:")
            new_plot_title = gr.Textbox(label="Plot title", placeholder="Optional (e.g. XYZ Comparison)")
            new_plot_vars = gr.Dropdown(choices=[], multiselect=True, label="Variables")
            with gr.Row():
                confirm_plot_btn = gr.Button("Generate Plot", variant="primary")
                cancel_plot_btn = gr.Button("Cancel")
                
        dynamic_plots_state = gr.State([])

        @gr.render(inputs=[dynamic_plots_state, ep_num_nb])
        def render_dynamic_plots(plots_config, ep_idx):
            for i, p_config in enumerate(plots_config):
                vars_list = p_config["vars"]
                title = p_config["title"]
                
                with gr.Group():
                    with gr.Row():
                        display_title = title if title else f"Custom Plot {i+1}"
                        gr.Markdown(f"**{display_title}**: {', '.join(vars_list)}")
                        export_html_btn = gr.DownloadButton("⬇️ HTML", size="sm", scale=0)
                        export_pdf_btn = gr.DownloadButton("⬇️ PDF", size="sm", scale=0)
                        del_btn = gr.Button("❌ Remove", size="sm", scale=0)
                    
                    fig = build_custom_plotly_chart(int(ep_idx), vars_list, title=display_title)
                    if fig:
                        gr.Plot(value=fig)
                    else:
                        gr.Markdown("*(No data or variables not available)*")

                    def make_export_html_fn(ep, v_list, t):
                        def export():
                            import tempfile
                            f = build_custom_plotly_chart(int(ep), v_list, title=t)
                            if not f: return None
                            path = tempfile.mktemp(suffix=".html", prefix="plot_")
                            f.write_html(path, include_plotlyjs='cdn')
                            return path
                        return export

                    export_html_btn.click(
                        fn=make_export_html_fn(ep_idx, vars_list, display_title),
                        inputs=[], outputs=[export_html_btn]
                    )

                    def make_export_pdf_fn(ep, v_list, t):
                        def export():
                            import tempfile
                            f = build_custom_plotly_chart(int(ep), v_list, title=t)
                            if not f: return None
                            path = tempfile.mktemp(suffix=".pdf", prefix="plot_")
                            try:
                                f.write_image(path, format="pdf")
                                return path
                            except Exception as e:
                                err = str(e)
                                # kaleido v1+ needs Chrome — try to auto-download it
                                if "Chrome" in err or "chrome" in err or "kaleido" in err.lower():
                                    try:
                                        import kaleido
                                        kaleido.get_chrome_sync()
                                        f.write_image(path, format="pdf")
                                        return path
                                    except Exception:
                                        pass
                                raise gr.Error(
                                    "PDF export requires Chrome for kaleido. "
                                    "Fix it by running in your terminal:\n\n"
                                    "    kaleido_get_chrome\n\n"
                                    "Or downgrade kaleido: pip install 'kaleido==0.2.1'"
                                )
                        return export

                    export_pdf_btn.click(
                        fn=make_export_pdf_fn(ep_idx, vars_list, display_title),
                        inputs=[], outputs=[export_pdf_btn]
                    )

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
        gr.Markdown("### Signals to include in the final dataset")

        with gr.Row(equal_height=True):
            with gr.Group():
                gr.Markdown("**Complete features** (cameras, action, observation.state)")
                with gr.Row():
                    whole_all_btn = gr.Button("All", size="sm", scale=1)
                    whole_none_btn = gr.Button("None", size="sm", scale=1)
                whole_search_tb = gr.Textbox(
                    placeholder="🔍 Search features...",
                    show_label=False, container=False,
                )
                whole_feature_cbg = gr.CheckboxGroup(
                    choices=[], value=[], show_label=False,
                    elem_classes=["scrollable-list"],
                )

            with gr.Group():
                gr.Markdown("**action** — dimensions")
                with gr.Row():
                    action_all_btn = gr.Button("All", size="sm", scale=1)
                    action_none_btn = gr.Button("None", size="sm", scale=1)
                action_search_tb = gr.Textbox(
                    placeholder="🔍 Search dimensions...",
                    show_label=False, container=False,
                )
                dims_action_cbg = gr.CheckboxGroup(
                    choices=[], value=[], show_label=False,
                    elem_classes=["scrollable-list"],
                )

            with gr.Group():
                gr.Markdown("**observation.state** — dimensions")
                with gr.Row():
                    state_all_btn = gr.Button("All", size="sm", scale=1)
                    state_none_btn = gr.Button("None", size="sm", scale=1)
                state_search_tb = gr.Textbox(
                    placeholder="🔍 Search dimensions...",
                    show_label=False, container=False,
                )
                dims_state_cbg = gr.CheckboxGroup(
                    choices=[], value=[], show_label=False,
                    elem_classes=["scrollable-list"],
                )

        gr.Markdown("---")
        gr.Markdown("### Language Instruction — global for the whole dataset")
        with gr.Row():
            task_tb = gr.Textbox(value="", show_label=False, interactive=False, scale=10)
            with gr.Column(scale=0, min_width=60):
                task_edit_btn = gr.Button("✏️", min_width=50)
                task_save_btn = gr.Button("💾", min_width=50, visible=False)
        task_dirty_state = gr.State(False)

        gr.Markdown("---")
        gr.Markdown("### Packaging")
        with gr.Row():
            new_repo_id_tb = gr.Textbox(value="", label="Final Repo ID (local)")
            curated_root_tb = gr.Textbox(value=str(HUB_ROOT.parent / "data_curated"), label="Root folder for the curated dataset")
        package_btn = gr.Button("Package dataset", variant="primary", size="lg")
        package_status = gr.Textbox(label="Packaging status", interactive=False)
        curation_log_box = gr.Textbox(label="Curation log", lines=10, max_lines=10, interactive=False)

        gr.Markdown("---")
        gr.Markdown(
            "### Upload to Hugging Face Hub\n"
            "You must package a dataset first (section above) before uploading it."
        )
        with gr.Accordion("❓ How do I get my Hugging Face token", open=False):
            gr.Markdown(
                "- Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) "
                "and create a new token with **Write** permission (or, if using a *fine-grained* token, "
                "enable write permission on dataset repos).\n"
                "- Copy the token (starts with `hf_...`) and paste it in the field below. "
                "It is not saved to disk, it is only used for this upload.\n"
                "- The **Repo ID** must start with your real Hugging Face username — check it at "
                "[huggingface.co/settings/profile](https://huggingface.co/settings/profile), field "
                "\"Username\" — or an organization you belong to with write access. "
                "Do not use your OS username or `local/...`: if it doesn't match your Hugging Face account "
                "you'll get a 403 Forbidden error."
            )
        with gr.Row():
            hf_repo_id_tb = gr.Textbox(
                label="Repo ID on Hugging Face",
                placeholder="your_user/dataset_name",
                scale=3,
            )
            hf_token_tb = gr.Textbox(
                label="Hugging Face Token (session only, not saved to disk)",
                type="password", placeholder="hf_...",
                scale=3,
            )
            hf_private_cb = gr.Checkbox(value=True, label="Private", scale=1)
        push_btn = gr.Button("Upload to Hugging Face", variant="stop", size="lg")
        push_status = gr.Textbox(label="Upload status", interactive=False)

    return dict(
        viz_root_tb=viz_root_tb, refresh_btn=refresh_btn,
        dataset_dd=dataset_dd, last_btn=last_btn,
        load_ds_btn=load_ds_btn, viz_status=viz_status,
        ep_prev_btn=ep_prev_btn, ep_num_nb=ep_num_nb, ep_next_btn=ep_next_btn,
        ep_delete_btn=ep_delete_btn, ep_info_tb=ep_info_tb, marked_summary_tb=marked_summary_tb,
        vid0=vid0, vid1=vid1, vid2=vid2,
        add_plot_btn=add_plot_btn, new_plot_group=new_plot_group, new_plot_vars=new_plot_vars,
        new_plot_title=new_plot_title,
        confirm_plot_btn=confirm_plot_btn, cancel_plot_btn=cancel_plot_btn,
        dynamic_plots_state=dynamic_plots_state,
        whole_all_btn=whole_all_btn, whole_none_btn=whole_none_btn, whole_feature_cbg=whole_feature_cbg,
        whole_search_tb=whole_search_tb,
        action_all_btn=action_all_btn, action_none_btn=action_none_btn, dims_action_cbg=dims_action_cbg,
        action_search_tb=action_search_tb,
        state_all_btn=state_all_btn, state_none_btn=state_none_btn, dims_state_cbg=dims_state_cbg,
        state_search_tb=state_search_tb,
        task_edit_btn=task_edit_btn, task_save_btn=task_save_btn,
        task_tb=task_tb, task_dirty_state=task_dirty_state,
        new_repo_id_tb=new_repo_id_tb, curated_root_tb=curated_root_tb,
        package_btn=package_btn, package_status=package_status, curation_log_box=curation_log_box,
        hf_repo_id_tb=hf_repo_id_tb, hf_token_tb=hf_token_tb, hf_private_cb=hf_private_cb,
        push_btn=push_btn, push_status=push_status,
    )
