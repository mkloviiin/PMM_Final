"""Vista en vivo de Episode Control: las cámaras de la sesión, en un solo stream MJPEG.

Fuentes (ZMQ PUB con JPEG, puertos en interfaces/zmq_image_publisher.py):
- sesión con VR: lo mismo que va a los lentes (ojo del robot con el overlay del VR) y la
  pantalla chica frontal (VR_STREAM_PORTS);
- sin VR: las cámaras que graba play.py, ojo del robot y frontal / cenital (PREVIEW_PORTS).

Las dos vistas se juntan lado a lado en un solo stream (ROUTE), servido por el mismo servidor
de Gradio: la página abre una sola conexión y funciona igual si el Hub se abre desde otro equipo.
Solo se decodifica / recodifica cuando llega una imagen nueva y alguien está mirando.
"""

from __future__ import annotations

import asyncio
import threading
import time

import cv2
import numpy as np

from interfaces.zmq_image_publisher import PREVIEW_PORTS, VR_STREAM_PORTS

from . import state

ROUTE = "/live/view.mjpg"
_VIEW_H = 360           # alto de cada vista en el cuadro combinado [px]
_STALE_S = 1.0          # sin imágenes nuevas en este tiempo: "No video"
_PERIOD_S = 1.0 / 15    # tope de cuadros por segundo del stream
_LABELS = {True: ("VR headset", "Front view"), False: ("Robot eye", "Scene camera")}

_frames: dict[int, tuple[bytes, float]] = {}   # puerto -> (último JPEG, hora de llegada)
_cache_lock = threading.Lock()
_cache: tuple[tuple, bytes] | None = None      # (clave de las fuentes, JPEG combinado)
_started = False


def _receive(port: int) -> None:
    """Hilo por puerto: guarda el último JPEG publicado (los PUB no guardan nada si no hay nadie)."""
    import zmq

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://127.0.0.1:{port}")
    while True:
        try:
            if sock.poll(500):
                parts = sock.recv_multipart()
                _frames[port] = (parts[-1], time.time())
        except Exception:  # noqa: BLE001
            time.sleep(0.5)


def _placeholder(text: str) -> np.ndarray:
    img = np.full((_VIEW_H, _VIEW_H * 4 // 3, 3), 40, dtype=np.uint8)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, ((img.shape[1] - w) // 2, (img.shape[0] + h) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)
    return img


def _view(port: int, label: str, now: float) -> np.ndarray:
    item = _frames.get(port)
    img = None
    if item is not None and now - item[1] <= _STALE_S:
        img = cv2.imdecode(np.frombuffer(item[0], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = _placeholder("No video")
    else:
        img = cv2.resize(img, (round(img.shape[1] * _VIEW_H / img.shape[0]), _VIEW_H),
                         interpolation=cv2.INTER_AREA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _combined() -> tuple[tuple, bytes]:
    """JPEG con las dos vistas de la sesión actual; se recalcula solo si cambió alguna fuente."""
    global _cache
    vr = bool(state.live_view_vr)
    ports = VR_STREAM_PORTS if vr else PREVIEW_PORTS
    now = time.time()
    # Clave: la imagen de cada fuente (su hora de llegada) o "sin video"
    key = (vr, *(t if (t := _frames.get(p, (b"", 0.0))[1]) and now - t <= _STALE_S else None
                 for p in (ports["robot"], ports["scene"])))
    with _cache_lock:
        if _cache is not None and _cache[0] == key:
            return _cache
        labels = _LABELS[vr]
        robot, scene = _view(ports["robot"], labels[0], now), _view(ports["scene"], labels[1], now)
        ok, buf = cv2.imencode(".jpg", np.hstack([robot, scene]), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        _cache = (key, buf.tobytes() if ok else b"")
        return _cache


async def _mjpeg():
    last = None
    while True:
        key, jpg = await asyncio.to_thread(_combined)
        if key != last and jpg:
            last = key
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() \
                + b"\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(_PERIOD_S)


def mount(app) -> None:
    """Agrega ROUTE al servidor de Gradio (después de launch) y arranca la recepción."""
    global _started
    from fastapi.responses import StreamingResponse

    app.add_api_route(
        ROUTE,
        lambda: StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
                                  headers={"Cache-Control": "no-store"}),
        methods=["GET"],
    )
    if not _started:
        _started = True
        for port in {*VR_STREAM_PORTS.values(), *PREVIEW_PORTS.values()}:
            threading.Thread(target=_receive, args=(port,), daemon=True, name=f"live_view_{port}").start()


def html() -> str:
    return (f'<img src="{ROUTE}" alt="Live view" '
            'style="width:100%;max-width:1280px;display:block;margin:auto;border-radius:6px;background:#282828">')
