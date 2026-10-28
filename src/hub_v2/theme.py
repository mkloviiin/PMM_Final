"""Tema, CSS y JS del Hub v2.

El CSS del Hub original (WELCOME_CSS) se incluye completo: los componentes de curación
reutilizados (selector de carpeta, listas con buscador, diálogo de borrado, reproductor de
videos) dependen de esas clases. Lo propio de la v2 va con prefijo .v2-.
"""

from __future__ import annotations

import gradio as gr

from launcher_gui.app import WELCOME_CSS

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.Font(f) for f in ("Inter", "ui-sans-serif", "system-ui", "sans-serif")],
    font_mono=[gr.themes.Font(f) for f in ("JetBrains Mono", "ui-monospace", "monospace")],
    radius_size=gr.themes.sizes.radius_md,
)

V2_CSS = """
:root {
    --v2-ok: #2f9e44;
    --v2-warn: #f59f00;
    --v2-err: #e03131;
    --v2-off: #868e96;
    --v2-rec: #e03131;
}
.gradio-container { width: 100% !important; max-width: 1500px !important; margin: auto !important; }
.v2-hidden { display: none !important; }

/* ── Barra superior: identidad + estado global, visible en todas las pestañas ── */
.v2-header {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    flex-wrap: wrap; padding: 10px 16px; border-radius: 12px;
    background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
}
.v2-brand { display: flex; align-items: center; gap: 14px; }
.v2-brand h1 { font-size: 1.25rem; margin: 0; font-weight: 700; letter-spacing: .2px; }
.v2-brand small { color: var(--body-text-color-subdued); font-size: .85rem; }
.v2-brand img { height: 30px; width: auto; }
.v2-chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.v2-chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
    font-size: .85rem; font-weight: 600; border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary); white-space: nowrap;
}
.v2-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--v2-off); flex: none; }
.v2-ok .v2-dot { background: var(--v2-ok); }
.v2-warn .v2-dot { background: var(--v2-warn); }
.v2-err .v2-dot { background: var(--v2-err); }
.v2-rec .v2-dot { background: var(--v2-rec); animation: v2-pulse 1s infinite; }
.v2-chip.v2-rec { border-color: var(--v2-rec); color: var(--v2-rec); }
.v2-chip.v2-err { border-color: var(--v2-err); color: var(--v2-err); }
@keyframes v2-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }

/* ── Tarjetas de sección ── */
.v2-card {
    border: 1px solid var(--border-color-primary) !important; border-radius: 12px !important;
    padding: 14px 16px !important; background: var(--block-background-fill) !important;
}
.v2-card-title p, .v2-card-title { margin: 0 0 4px 0 !important; font-weight: 700; font-size: 1.02rem; }
.v2-muted, .v2-muted p { color: var(--body-text-color-subdued) !important; font-size: .9rem; }

/* ── Plataforma: radio dibujado como tarjetas ── */
.v2-platform .wrap { display: grid !important; grid-template-columns: repeat(3, 1fr); gap: 10px !important; }
.v2-platform .wrap label {
    justify-content: center; padding: 16px 10px !important; border-radius: 10px !important;
    border: 2px solid var(--border-color-primary) !important; font-weight: 600; font-size: 1rem;
}
.v2-platform .wrap label:has(input:checked) {
    border-color: var(--color-accent) !important; background: var(--color-accent-soft) !important;
}
.v2-platform .wrap label input { display: none; }

/* ── Pasos del stack Gazebo / robot real ── */
.v2-step { border-left: 4px solid var(--v2-off) !important; }
.v2-step.v2-step-ok { border-left-color: var(--v2-ok) !important; }
.v2-step-status { display: flex; align-items: center; gap: 8px; font-size: .92rem; }
.v2-num {
    display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px;
    border-radius: 50%; background: var(--color-accent); color: #fff; font-size: .8rem;
    font-weight: 700; margin-right: 8px;
}

/* ── Resumen antes de lanzar ── */
.v2-summary { border: 1px dashed var(--color-accent) !important; }
.v2-summary blockquote { margin: 6px 0 !important; }

/* ── Panel de grabación ── */
.v2-state {
    border-radius: 12px; padding: 16px 18px; border: 2px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.v2-state .v2-state-title { font-size: 1.6rem; font-weight: 800; display: flex; align-items: center; gap: 10px; }
.v2-state .v2-dot { width: 16px; height: 16px; }
.v2-state .v2-state-hint { margin-top: 6px; color: var(--body-text-color-subdued); }
.v2-state.v2-rec { border-color: var(--v2-rec); }
.v2-state.v2-rec .v2-state-title { color: var(--v2-rec); }
.v2-state.v2-ok { border-color: var(--v2-ok); }
.v2-state.v2-warn { border-color: var(--v2-warn); }
.v2-state.v2-err { border-color: var(--v2-err); }
.v2-progress { margin-top: 4px; }
.v2-progress .v2-bar { height: 12px; border-radius: 6px; background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); overflow: hidden; }
.v2-progress .v2-fill { height: 100%; background: var(--v2-ok); transition: width .4s; }
.v2-progress .v2-progress-text { display: flex; justify-content: space-between; font-size: .9rem;
    margin-top: 4px; color: var(--body-text-color-subdued); }
.v2-big button, button.v2-big { min-height: 64px !important; font-size: 1.15rem !important; font-weight: 700 !important; }
.v2-kbd { font-family: var(--font-mono); border: 1px solid var(--border-color-primary); border-radius: 4px;
    padding: 0 5px; font-size: .8rem; }
.v2-cheatsheet table { font-size: .9rem; }

/* ── Revisión: estado vacío ── */
.v2-empty { text-align: center; padding: 40px 10px !important; color: var(--body-text-color-subdued); }
"""

CSS = WELCOME_CSS + V2_CSS

# Modo oscuro fijo (el laboratorio trabaja con luz baja) + atajos de teclado de Record:
# R = grabar, S = detener y guardar. Solo con la pestaña Record a la vista y sin un campo de
# texto enfocado (si no, escribir una nota dispararía la grabación).
JS = """
() => {
    document.body.classList.add('dark');
    if (window.__hubV2Keys) return;
    window.__hubV2Keys = true;
    document.addEventListener('keydown', e => {
        if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
        if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable="true"]')) return;
        const panel = document.getElementById('v2-record-panel');
        if (!panel || panel.offsetParent === null) return;
        const id = {r: 'v2-rec-btn', s: 'v2-stop-btn'}[e.key.toLowerCase()];
        const btn = id && document.getElementById(id);
        if (!btn) return;
        (btn.tagName === 'BUTTON' ? btn : btn.querySelector('button'))?.click();
        e.preventDefault();
    });
}
"""
