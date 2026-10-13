"""Videos de camara sincronizados con un controlador comun + cursor de tiempo
sobre los graficos.

- `build_synced_video_html` genera el bloque HTML con los videos del episodio
  (sin controles propios) y, debajo, una sola barra de control: arriba la
  barra de tiempo y, bajo ella, retroceder/avanzar 5 s, play/pausa, velocidad
  y tiempo/duracion.
- `SYNC_JS` se ejecuta una vez al cargar la pagina (`demo.load(js=...)`) y:
    * maneja esa barra de control (eventos delegados en `document`, porque
      gr.HTML no ejecuta <script> ni handlers inline);
    * usa el primer video como reloj y corrige el drift de los demas;
    * dibuja una linea vertical en cada grafico Plotly marcado con
      `layout.meta.video_sync` que sigue el tiempo del video (como el dataset
      visualizer de LeRobot);
    * al hacer click sobre un grafico, mueve los videos a ese instante.
"""

from __future__ import annotations

import html
from pathlib import Path


def _video_src(path: str | None) -> str | None:
    if not path:
        return None
    if not Path(path).exists():
        return None
    return f"/gradio_api/file={path}"


SKIP_SECONDS = 5

# Iconos SVG (viewBox 24x24); el color lo pone PLAYER_CSS.
_ICON_PLAY = "<svg class='vs-ico-play' viewBox='0 0 24 24'><path d='M8 5v14l11-7z'/></svg>"
_ICON_PAUSE = "<svg class='vs-ico-pause' viewBox='0 0 24 24'><path d='M6 5h4v14H6zM14 5h4v14h-4z'/></svg>"
_ICON_REWIND = "<svg viewBox='0 0 24 24'><path d='M11 18V6l-8.5 6zM20 18V6l-8.5 6z'/></svg>"
_ICON_FORWARD = "<svg viewBox='0 0 24 24'><path d='M4 18l8.5-6L4 6zM13 18l8.5-6L13 6z'/></svg>"

# Se agrega al css de la app (ver app.py). Colores fijos (el tema usa rojo como
# primario) y con !important porque el tema pisa color/fill de botones y svg.
PLAYER_CSS = """
.vs-player .vs-btn,
.vs-player .vs-rate {
    height: 36px;
    border: none;
    border-radius: 8px;
    background: #f97316 !important;
    color: #fff !important;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
}
.vs-player .vs-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    min-width: 52px;
    padding: 0 12px;
}
.vs-player .vs-btn:hover,
.vs-player .vs-rate:hover {
    background: #ea580c !important;
}
.vs-player .vs-btn svg,
.vs-player .vs-btn svg path {
    width: 20px;
    height: 20px;
    fill: #fff !important;
}
.vs-player .vs-btn[data-state="playing"] .vs-ico-play,
.vs-player .vs-btn[data-state="paused"] .vs-ico-pause {
    display: none;
}
.vs-player .vs-rate {
    padding: 0 8px;
}
.vs-player .vs-rate option {
    color: #000;
    background: #fff;
}
"""


def build_synced_video_html(v0: str | None, v1: str | None, v2: str | None,
                            labels: list[str] | None = None) -> str:
    srcs = [_video_src(v0), _video_src(v1), _video_src(v2)]
    labels = (list(labels or []) + ["Camera 0", "Camera 1", "Camera 2"][len(labels or []):])[:3]

    if not any(srcs):
        return "<div style='padding:20px;text-align:center;color:#888;'>No videos for this episode.</div>"

    tags = []
    for i, (src, label) in enumerate(zip(srcs, labels)):
        if src:
            tags.append(
                f"<div style='flex:1;min-width:0;'>"
                f"<div style='text-align:center;font-size:0.85rem;margin-bottom:4px;'>{html.escape(label)}</div>"
                f"<video id='sync-vid-{i}' src='{html.escape(src, quote=True)}' "
                f"autoplay muted playsinline preload='auto' "
                f"style='width:100%;border-radius:8px;display:block;'></video>"
                f"</div>"
            )
        else:
            tags.append("<div style='flex:1;min-width:0;'></div>")

    # Fila 1: barra de tiempo a todo el ancho.
    # Fila 2: botones + velocidad al centro, tiempo/duracion a la derecha.
    controls = (
        "<input type='range' data-vs='seek' min='0' max='1' step='any' value='0' "
        "style='display:block;width:100%;margin-top:10px;"
        "accent-color:#f97316;cursor:pointer;'>"
        "<div style='display:grid;grid-template-columns:1fr auto 1fr;align-items:center;margin-top:6px;'>"
        "<div></div>"
        "<div style='display:flex;align-items:center;gap:8px;'>"
        f"<button type='button' class='vs-btn' data-vs='skip' data-sec='-{SKIP_SECONDS}' "
        f"title='Back {SKIP_SECONDS} s'>{_ICON_REWIND}{SKIP_SECONDS}s</button>"
        f"<button type='button' class='vs-btn' data-vs='play' data-state='playing' "
        f"title='Play / pause'>{_ICON_PLAY}{_ICON_PAUSE}</button>"
        f"<button type='button' class='vs-btn' data-vs='skip' data-sec='{SKIP_SECONDS}' "
        f"title='Forward {SKIP_SECONDS} s'>{SKIP_SECONDS}s{_ICON_FORWARD}</button>"
        "<select class='vs-rate' data-vs='rate' title='Playback speed'>"
        "<option value='0.25'>0.25x</option><option value='0.5'>0.5x</option>"
        "<option value='1' selected>1x</option><option value='2'>2x</option>"
        "</select>"
        "</div>"
        "<span data-vs='time' style='justify-self:end;font-family:monospace;font-size:0.9rem;"
        "white-space:nowrap;'>0.00 / 0.00 s</span>"
        "</div>"
    )

    return ("<div class='vs-player'>"
            "<div style='display:flex;gap:12px;'>" + "".join(tags) + "</div>"
            + controls + "</div>")


SYNC_JS = """
() => {
    if (window.__icubVideoSync) return;
    window.__icubVideoSync = true;

    const VIDEO_SEL = 'video[id^="sync-vid-"]';
    const CURSOR_COLOR = '#ff5252';
    const DRIFT_TOL = 0.1;   // s: desfase maximo tolerado entre camaras

    const videos = () => Array.from(document.querySelectorAll(VIDEO_SEL));
    // El primer video es el reloj: barra de control y cursor leen su tiempo.
    const master = () => videos()[0] || null;
    const player = () => document.querySelector('.vs-player');
    const ctl = name => { const p = player(); return p && p.querySelector('[data-vs="' + name + '"]'); };

    function seekAll(t) {
        const m = master();
        if (!m) return;
        if (isFinite(m.duration)) t = Math.min(t, m.duration - 0.001);
        t = Math.max(0, t);
        videos().forEach(v => { v.currentTime = t; });
    }
    function playAll() {
        const m = master();
        if (!m) return;
        if (m.ended) seekAll(0);
        videos().forEach(v => {
            if (v !== m) v.currentTime = m.currentTime;
            v.play().catch(() => {});
        });
    }
    function pauseAll() {
        const m = master();
        if (!m) return;
        videos().forEach(v => v.pause());
        videos().forEach(v => { if (v !== m) v.currentTime = m.currentTime; });
    }

    // ── Barra de control comun (eventos delegados) ──────────────────────
    let scrubbing = false;
    document.addEventListener('click', e => {
        const b = e.target.closest && e.target.closest('.vs-player [data-vs]');
        if (!b) return;
        const m = master();
        if (!m) return;
        if (b.dataset.vs === 'play') {
            (m.paused || m.ended) ? playAll() : pauseAll();
        } else if (b.dataset.vs === 'skip') {
            seekAll(m.currentTime + Number(b.dataset.sec));
        }
    });
    document.addEventListener('pointerdown', e => {
        if (e.target.matches && e.target.matches('.vs-player [data-vs="seek"]')) scrubbing = true;
    }, true);
    document.addEventListener('pointerup', () => { scrubbing = false; }, true);
    document.addEventListener('input', e => {
        if (e.target.matches && e.target.matches('.vs-player [data-vs="seek"]')) {
            seekAll(Number(e.target.value));
        }
    }, true);
    document.addEventListener('change', e => {
        if (e.target.matches && e.target.matches('.vs-player [data-vs="rate"]')) {
            videos().forEach(v => { v.playbackRate = Number(e.target.value) || 1; });
        }
    }, true);
    // Loop: cuando el reloj termina, todos vuelven al inicio juntos.
    document.addEventListener('ended', e => {
        if (e.target === master()) { seekAll(0); playAll(); }
    }, true);

    // ── Geometria de un grafico Plotly sincronizable ────────────────────
    function plotGeom(gd) {
        const fl = gd._fullLayout;
        const meta = (gd.layout && gd.layout.meta) || (fl && fl.meta);
        if (!fl || !meta || !meta.video_sync || !fl.xaxis || !fl._size) return null;
        const x0 = Number(fl.xaxis.range[0]), x1 = Number(fl.xaxis.range[1]);
        if (!(x1 > x0)) return null;
        return { x0, x1, s: fl._size, t0: Number(meta.t0) || 0 };
    }

    function cursorFor(gd) {
        let c = gd.querySelector(':scope > .vs-cursor');
        if (!c) {
            if (getComputedStyle(gd).position === 'static') gd.style.position = 'relative';
            c = document.createElement('div');
            c.className = 'vs-cursor';
            c.style.cssText = 'position:absolute;width:2px;margin-left:-1px;pointer-events:none;'
                + 'z-index:10;display:none;background:' + CURSOR_COLOR + ';';
            const lbl = document.createElement('span');
            lbl.style.cssText = 'position:absolute;top:-16px;left:4px;font:11px monospace;'
                + 'white-space:nowrap;color:' + CURSOR_COLOR + ';';
            c.appendChild(lbl);
            gd.appendChild(c);
        }
        return c;
    }

    // ── Bucle: barra de control, cursor en graficos y drift ─────────────
    function tick() {
        const m = master();

        if (m) {
            const rate = ctl('rate');
            const r = rate ? Number(rate.value) || 1 : 1;
            videos().forEach(v => { if (v.playbackRate !== r) v.playbackRate = r; });

            const seek = ctl('seek');
            if (seek && isFinite(m.duration)) {
                seek.max = m.duration;
                if (!scrubbing) seek.value = m.currentTime;
            }
            const time = ctl('time');
            if (time) {
                const dur = isFinite(m.duration) ? m.duration : 0;
                time.textContent = m.currentTime.toFixed(2) + ' / ' + dur.toFixed(2) + ' s';
            }
            const play = ctl('play');
            if (play) {
                const st = (m.paused || m.ended) ? 'paused' : 'playing';
                if (play.dataset.state !== st) play.dataset.state = st;
            }

            if (!m.paused && !m.seeking) {
                videos().forEach(v => {
                    if (v === m || v.seeking) return;
                    if (v.paused) v.play().catch(() => {});
                    if (Math.abs(v.currentTime - m.currentTime) > DRIFT_TOL) v.currentTime = m.currentTime;
                });
            }
        }

        document.querySelectorAll('.js-plotly-plot').forEach(gd => {
            const g = plotGeom(gd);
            if (!g) return;
            const c = cursorFor(gd);
            const x = m ? g.t0 + m.currentTime : NaN;
            if (!isFinite(x) || x < g.x0 || x > g.x1) { c.style.display = 'none'; return; }
            c.style.display = 'block';
            c.style.left = (g.s.l + (x - g.x0) / (g.x1 - g.x0) * g.s.w) + 'px';
            c.style.top = g.s.t + 'px';
            c.style.height = g.s.h + 'px';
            c.firstChild.textContent = m.currentTime.toFixed(2) + ' s';
        });

        requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    // ── Click en un grafico -> mover los videos a ese instante ──────────
    // Se ignoran los arrastres (zoom de Plotly) comparando con el mousedown.
    let downAt = null;
    document.addEventListener('mousedown', e => { downAt = [e.clientX, e.clientY]; }, true);
    document.addEventListener('click', e => {
        if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
        const gd = e.target.closest && e.target.closest('.js-plotly-plot');
        if (!gd || !master()) return;
        const g = plotGeom(gd);
        if (!g) return;
        const r = gd.getBoundingClientRect();
        const px = e.clientX - r.left - g.s.l, py = e.clientY - r.top - g.s.t;
        if (px < 0 || px > g.s.w || py < 0 || py > g.s.h) return;
        seekAll(g.x0 + px / g.s.w * (g.x1 - g.x0) - g.t0);
    }, true);
}
"""
