"""Variantes de la escena de la puerta (scene_icub_door.xml) sin tocar el XML.

- set_handle_side: manilla a la izquierda del robot (bisagra a la derecha, como el XML) o a la
  derecha (bisagra a la izquierda). Espeja, en el plano de la hoja, la bisagra y el cuerpo de la
  manilla; door_hinge > 0 sigue siendo "abierta" y handle_hinge < 0 "bajada".
- place_door: mueve todo el montaje para que el centro de la hoja quede en (x, y) del mundo.

Solo se cambian poses de cuerpos y de joints: MuJoCo precalcula las cajas de colisión de cada
cuerpo con sus geoms al compilar, así que mover un geom dentro de su cuerpo dejaría de chocar.
Por eso el marco trae las dos trabas (latch_strike y latch_strike_mirror) y la manilla se
espeja girando su cuerpo 180° alrededor del eje y de la hoja (normal a la hoja).
Los cambios son del MjModel (no del MjData): después hay que llamar a mj_forward.
"""

from __future__ import annotations

import numpy as np

# Giro de 180° alrededor de y del marco de la hoja: x -> -x, z -> -z. La manilla y el
# pestillo son simétricos en z, así que equivale a espejarlos en x.
_QUAT_FLIP_Y = np.array([0.0, 0.0, 1.0, 0.0])


def handle_side(model) -> str:
    """'right' si la manilla queda a la derecha del robot (y < 0), 'left' si no."""
    # Bisagra en x<0 del marco de la hoja = lado +y del mundo = manilla a la derecha
    return "right" if model.jnt_pos[model.joint("door_hinge").id][0] < 0 else "left"


def _mirror_axis(axis) -> np.ndarray:
    # Espejo x -> -x de un eje de giro (pseudovector)
    return np.array([axis[0], -axis[1], -axis[2]])


def _quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def set_handle_side(model, side: str) -> None:
    """Cambiar de lado es su propia inversa: se aplica el mismo espejo en ambos sentidos."""
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    if handle_side(model) == side:
        return
    hinge = model.joint("door_hinge").id
    handle = model.joint("handle_hinge").id
    latch = model.body("latch").id
    panel = model.body("door_panel").id
    model.jnt_pos[hinge][0] *= -1
    model.jnt_axis[hinge] = _mirror_axis(model.jnt_axis[hinge])
    model.body_ipos[panel][0] *= -1
    # Cuerpo de la manilla: posición espejada y girado 180° en y. Su eje (en el marco del
    # cuerpo) se espeja y se expresa en el cuerpo girado: (x, y, z) -> (-x, y, -z).
    model.body_pos[latch][0] *= -1
    model.body_quat[latch] = _quat_mul(model.body_quat[latch], _QUAT_FLIP_Y)
    ax = _mirror_axis(model.jnt_axis[handle])
    model.jnt_axis[handle] = np.array([-ax[0], ax[1], -ax[2]])


def place_door(model, data, panel_xy) -> None:
    """Centro de la hoja en panel_xy (mundo). Usa data con la cinemática al día."""
    mount = model.body("door_mount").id
    panel_now = data.xpos[model.body("door_panel").id][:2]
    model.body_pos[mount][:2] += np.asarray(panel_xy, dtype=float) - panel_now
