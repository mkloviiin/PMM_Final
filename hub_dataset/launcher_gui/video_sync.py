"""Genera el bloque HTML con los 3 videos de camara sincronizados
(play/pause/seek en uno se replica en los otros dos)."""

from __future__ import annotations

from pathlib import Path

def _video_src(path: str | None) -> str | None:
    if not path:
        return None
    if not Path(path).exists():
        return None
    return f"/gradio_api/file={path}"

def build_synced_video_html(v0: str | None, v1: str | None, v2: str | None) -> str:
    srcs = [_video_src(v0), _video_src(v1), _video_src(v2)]
    labels = ["Camera 0", "Camera 1", "Camera 2"]

    if not any(srcs):
        return "<div style='padding:20px;text-align:center;color:#888;'>No videos for this episode.</div>"

    tags = []
    for i, (src, label) in enumerate(zip(srcs, labels)):
        if src:
            tags.append(
                f"<div style='flex:1;min-width:0;'>"
                f"<div style='text-align:center;font-size:0.85rem;margin-bottom:4px;'>{label}</div>"
                f"<video id='sync-vid-{i}' data-sync-group='ep-videos' "
                f"src='{src}' controls loop autoplay muted "
                f"style='width:100%;border-radius:8px;'></video>"
                f"</div>"
            )
        else:
            tags.append("<div style='flex:1;min-width:0;'></div>")

    return "<div style='display:flex;gap:12px;'>" + "".join(tags) + "</div>"