"""Help overlay drawn on the big VR screen, shared by teleop_mujoco.py (MuJoCo) and
teleop_module_sm.py (Gazebo / real robot)."""

from __future__ import annotations

import cv2
import numpy as np

# VR help on the big screen: small "X: Ayuda" in the bottom-right corner and, while the app is
# paused (or after pressing X, if BeaVR sends it to the PC), a panel with the buttons.
# No accents: cv2.putText does not support them.
VR_HELP_HINT = "X: Ayuda"
VR_HELP_LINES = (
    ("A", "Empezar a grabar"),
    ("B", "Guardar episodio"),
    ("B (mantener 1.5 s)", "Descartar episodio"),
    ("Y", "Reiniciar escena"),
    ("Gatillo indice", "Abrir / cerrar mano"),
    ("Gatillo inferior", "Embrague: acomodar el control"),
    ("X", "Pausa / ayuda"),
)
# No controller data for longer than this [s] = the app is paused (BeaVR stops sending them;
# the PC keeps sending the image).
VR_PAUSE_TIMEOUT_S = 0.3


def draw_help_hint(img: np.ndarray) -> np.ndarray:
    """Draw (in place) a small semi-transparent "X: Ayuda" in the bottom-right corner of BGR `img`."""
    h, w = img.shape[:2]
    k = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc, thick = 0.5 * k, max(1, round(1 * k))
    pad, margin = max(3, int(6 * k)), max(4, int(14 * k))
    (tw, th), base = cv2.getTextSize(VR_HELP_HINT, font, sc, thick)
    x1, y1 = w - margin, h - margin
    x0, y0 = max(0, x1 - tw - 2 * pad), max(0, y1 - th - base - 2 * pad)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.6).astype(img.dtype)
    text = img[y0:y1, x0:x1].copy()
    cv2.putText(text, VR_HELP_HINT, (pad, pad + th), font, sc, (255, 255, 255), thick, cv2.LINE_AA)
    img[y0:y1, x0:x1] = cv2.addWeighted(text, 0.7, img[y0:y1, x0:x1], 0.3, 0)
    return img


def draw_help_panel(img: np.ndarray, title: str = "AYUDA") -> np.ndarray:
    """Draw (in place) a centered panel with the VR buttons (VR_HELP_LINES) on BGR `img`."""
    h, w = img.shape[:2]
    k = w / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, round(1 * k))
    sc_title, sc = 0.8 * k, 0.5 * k
    pad, gap = max(6, int(16 * k)), max(4, int(10 * k))
    (tw_t, th_t), _ = cv2.getTextSize(title, font, sc_title, thick + 1)
    sizes_k = [cv2.getTextSize(key, font, sc, thick)[0] for key, _ in VR_HELP_LINES]
    sizes_d = [cv2.getTextSize(desc, font, sc, thick)[0] for _, desc in VR_HELP_LINES]
    col_k = max(tw for tw, _ in sizes_k)
    row_h = max(th for _, th in sizes_k + sizes_d) + gap
    box_w = max(tw_t, col_k + 2 * gap + max(tw for tw, _ in sizes_d)) + 2 * pad
    box_h = th_t + gap + row_h * len(VR_HELP_LINES) + 2 * pad
    x0, y0 = max(0, (w - box_w) // 2), max(0, (h - box_h) // 2)
    x1, y1 = min(w, x0 + box_w), min(h, y0 + box_h)
    img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.25).astype(img.dtype)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (200, 200, 200), thick, cv2.LINE_AA)
    y = y0 + pad + th_t
    cv2.putText(img, title, (x0 + (box_w - tw_t) // 2, y), font, sc_title, (0, 220, 255), thick + 1, cv2.LINE_AA)
    y += gap
    for key, desc in VR_HELP_LINES:
        y += row_h
        cv2.putText(img, key, (x0 + pad, y), font, sc, (0, 220, 255), thick, cv2.LINE_AA)
        cv2.putText(img, desc, (x0 + pad + col_k + 2 * gap, y), font, sc, (255, 255, 255), thick, cv2.LINE_AA)
    return img
