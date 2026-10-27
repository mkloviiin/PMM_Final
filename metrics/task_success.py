"""Criterios de éxito de las 6 tareas, portados de robosuite (`_check_success` de cada entorno
en robosuite/environments/manipulation/, versión 1.5.2 de icubenv).

Lo usa eval_policy.py (MuJoCo y Gazebo): cada simulador arma un SceneState (poses de los
objetos, contactos, articulaciones) y aquí se decide el éxito con el mismo código. Como en los
rollouts de robomimic, el episodio es un éxito si el criterio se cumple en algún paso; los
evaluadores también guardan el estado final, porque un objeto puede cumplirlo un instante y
caerse después.

Posiciones con z = altura en el mundo (mesa en table_z, 0.72 m en ambos simuladores); x/y y
las orientaciones pueden estar en cualquier frame común a todos los objetos (mundo MuJoCo, mundo
Gazebo o root): los criterios solo usan alturas y poses relativas entre objetos.

Agarre (`_check_grasp` de robosuite: contacto de las dos almohadillas de la pinza): en el iCub,
la yema del pulgar y la del índice o el medio tocando el objeto, en cualquiera de las dos manos.
Son las yemas con sensor de contacto en Gazebo (model.urdf, /icub/touch/<mano>_<dedo>); MuJoCo
usa los mismos eslabones (r_hand_thumb_3, r_hand_index_3, r_hand_middle_3 y los de la izquierda).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TABLE_Z = 0.72

# Escena (XML de MuJoCo, también `file` en scenes.yaml) -> tarea
TASK_BY_SCENE = {
    "scene_icub_lift.xml": "lift",
    "scene_icub_stack.xml": "stack",
    "scene_icub_pickplace.xml": "pickplace",
    "scene_icub_door.xml": "door",
    "scene_icub_twoarmlift.xml": "twoarmlift",
    "scene_icub_peginhole.xml": "peginhole",
}

# Lo que cada criterio necesita del simulador
BODIES = {
    "lift": ["blue-cube"],
    "stack": ["blue-cube", "red-cube"],
    "pickplace": ["pick-cube", "place-bin"],
    "door": [],
    "twoarmlift": ["lift-pot"],
    "peginhole": ["peg", "hole-board"],
}
GRASP_BODIES = {"stack": ["blue-cube"], "pickplace": ["pick-cube"]}
CONTACT_PAIRS = {"stack": [("blue-cube", "red-cube")]}
JOINTS = {"door": ["door_hinge"]}

# Yemas que cuentan para el agarre: (pulgar, dedos opuestos)
GRASP_THUMB = "thumb"
GRASP_FINGERS = ("index", "middle")

# Lift._check_success / Stack.staged_rewards: centro del cubo sobre la mesa + 4 cm
LIFT_MARGIN_M = 0.04
# Door._check_success: bisagra sobre 0.3 rad
DOOR_HINGE_RAD = 0.3
# TwoArmLift._check_success: fondo de la olla sobre la mesa + 10 cm. El fondo de lift-pot está
# 6 cm bajo su origen (base de 1 cm en z = -0.055, scene_icub_twoarmlift.xml / models/lift-pot)
TWOARMLIFT_MARGIN_M = 0.10
POT_BOTTOM_OFFSET_M = 0.06
# PickPlace.not_in_bin: centro del objeto dentro del contenedor en xy y entre su base y 10 cm
# más arriba. place-bin mide 12 x 12 cm (semiancho 0.06) y su origen es el centro de la base.
BIN_HALF_XY_M = 0.06
BIN_HEIGHT_M = 0.10
# TwoArmPegInHole._check_success: d < 0.06, -0.12 <= t <= 0.14, cos > 0.95. La clavija mide
# 0.13 m como en robosuite (t y cos sin cambios); el agujero es de 6 x 6 cm y no de 10 x 10, así
# que d se escala con él: 0.06 * 0.03 / 0.05 (con 0.06 el eje podría pasar fuera del agujero).
PEG_D_MAX_M = 0.06 * 0.03 / 0.05
PEG_T_RANGE_M = (-0.12, 0.14)
PEG_COS_MIN = 0.95
# El agujero está en el origen de hole-board y su eje es el y local (tabla en el plano x-z);
# la clavija es un cilindro a lo largo de su z local (como en robosuite)
HOLE_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
PEG_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])


def quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float) / max(float(np.linalg.norm(quat)), 1e-9)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class SceneState:
    """poses: cuerpo -> (pos, quat wxyz); grasped: cuerpos tomados por una mano; touching:
    pares de cuerpos en contacto; joints: articulación -> ángulo [rad]."""
    poses: dict = field(default_factory=dict)
    table_z: float = TABLE_Z
    grasped: set = field(default_factory=set)
    touching: set = field(default_factory=set)
    joints: dict = field(default_factory=dict)

    def pos(self, body: str) -> np.ndarray:
        return np.asarray(self.poses[body][0], dtype=float)

    def mat(self, body: str) -> np.ndarray:
        return quat_to_mat(self.poses[body][1])

    def in_contact(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.touching


def _lift(s: SceneState) -> dict:
    z = s.pos("blue-cube")[2]
    return {"success": bool(z > s.table_z + LIFT_MARGIN_M), "cube_above_table": round(float(z - s.table_z), 4)}


def _stack(s: SceneState) -> dict:
    blue, red = s.pos("blue-cube"), s.pos("red-cube")
    lifted = bool(blue[2] > s.table_z + LIFT_MARGIN_M)
    touching = s.in_contact("blue-cube", "red-cube")
    grasped = "blue-cube" in s.grasped
    return {"success": lifted and touching and not grasped, "lifted": lifted, "touching": touching,
            "grasped": grasped, "blue_minus_red": (blue - red).round(3).tolist()}


def _pickplace(s: SceneState) -> dict:
    obj, bin_pos = s.pos("pick-cube"), s.pos("place-bin")
    local = s.mat("place-bin").T @ (obj - bin_pos)
    in_bin = bool(abs(local[0]) < BIN_HALF_XY_M and abs(local[1]) < BIN_HALF_XY_M
                  and bin_pos[2] < obj[2] < bin_pos[2] + BIN_HEIGHT_M)
    grasped = "pick-cube" in s.grasped
    return {"success": in_bin and not grasped, "in_bin": in_bin, "grasped": grasped,
            "obj_in_bin_frame": local.round(3).tolist()}


def _door(s: SceneState) -> dict:
    hinge = float(s.joints.get("door_hinge", 0.0))
    return {"success": hinge > DOOR_HINGE_RAD, "door_deg": round(float(np.degrees(hinge)), 1)}


def _twoarmlift(s: SceneState) -> dict:
    bottom = s.pos("lift-pot")[2] - POT_BOTTOM_OFFSET_M
    return {"success": bool(bottom > s.table_z + TWOARMLIFT_MARGIN_M),
            "pot_bottom_above_table": round(float(bottom - s.table_z), 4)}


def peg_hole_geometry(s: SceneState) -> tuple[float, float, float]:
    """TwoArmPegInHole._compute_orientation: (t, d, cos) entre el eje de la clavija y el agujero."""
    peg_pos, center = s.pos("peg"), s.pos("hole-board")
    v = s.mat("peg") @ PEG_AXIS_LOCAL
    v = v / np.linalg.norm(v)
    t = float((center - peg_pos) @ v)
    d = float(np.linalg.norm(np.cross(v, peg_pos - center)))
    normal = s.mat("hole-board") @ HOLE_AXIS_LOCAL
    cos = float(abs(normal @ v) / np.linalg.norm(normal))
    return t, d, cos


def _peginhole(s: SceneState) -> dict:
    t, d, cos = peg_hole_geometry(s)
    ok = d < PEG_D_MAX_M and PEG_T_RANGE_M[0] <= t <= PEG_T_RANGE_M[1] and cos > PEG_COS_MIN
    return {"success": bool(ok), "t": round(t, 4), "d": round(d, 4), "cos": round(cos, 4)}


CHECKS = {"lift": _lift, "stack": _stack, "pickplace": _pickplace, "door": _door,
          "twoarmlift": _twoarmlift, "peginhole": _peginhole}

DESCRIPTION = {
    "lift": f"blue cube center > table + {LIFT_MARGIN_M} m (robosuite Lift)",
    "stack": f"blue cube > table + {LIFT_MARGIN_M} m, touching the red cube and not grasped (robosuite Stack)",
    "pickplace": f"object inside the bin (±{BIN_HALF_XY_M} m, base to +{BIN_HEIGHT_M} m) and not grasped "
                 "(robosuite PickPlace)",
    "door": f"door_hinge > {DOOR_HINGE_RAD} rad (robosuite Door)",
    "twoarmlift": f"pot bottom > table + {TWOARMLIFT_MARGIN_M} m (robosuite TwoArmLift)",
    "peginhole": f"peg axis d < {PEG_D_MAX_M:.3f} m from the hole, {PEG_T_RANGE_M[0]} <= t <= "
                 f"{PEG_T_RANGE_M[1]} m, cos > {PEG_COS_MIN} (robosuite TwoArmPegInHole, d scaled to the hole)",
}


def check(task: str, s: SceneState) -> dict:
    """{"success": bool, ...} del criterio de `task` sobre el estado s."""
    return CHECKS[task](s)


class EpisodeSuccess:
    """Acumula el criterio paso a paso: éxito si se cumple en algún paso (robomimic) y estado final."""

    def __init__(self, task: str):
        self.task = task
        self.first_step: int | None = None
        self.last: dict = {}

    def update(self, step: int, s: SceneState) -> dict:
        self.last = check(self.task, s)
        if self.first_step is None and self.last["success"]:
            self.first_step = step
        return self.last

    @property
    def success(self) -> bool:
        return self.first_step is not None

    def summary(self) -> dict:
        return {"success": self.success, "success_step": self.first_step,
                "success_at_end": bool(self.last.get("success", False)),
                **{f"final_{k}": v for k, v in self.last.items() if k != "success"}}
