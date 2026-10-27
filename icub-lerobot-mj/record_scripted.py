#!/usr/bin/env python3
"""Graba datasets de las tareas de MuJoCo sin teleoperar: la trayectoria de las manos la genera
un plan por tarea en vez del mouse o el VR.

Usa el mismo robot que el Hub (lerobot_robot_icub, iCubMuJoCo), las mismas features, el mismo
frame root, las cámaras de la tarea (cameras en scenes.yaml) y el mismo formato LeRobot que
_manual_vr_record de play_mujoco.py: el dataset se entrena y evalúa igual que uno teleoperado.
Solo cambia de dónde sale la acción (rh_/lh_ pos, quat y gripper, gaze), que se graba tal cual
se le envía al robot.

Cada episodio parte con el robot en home, las manos abiertas y los objetos donde los pone el
reset de la escena (rangos de scenes.yaml). Solo se guardan los episodios en que la tarea
quedó cumplida (criterio de cada tarea); los demás se descartan y se reintenta.

Tareas: door (puerta con pestillo), stacking, pickplace, twoarmlift, peginhole.

Uso (entorno icubenv, el mismo del Hub):
    python record_scripted.py door --num-episodes 50 --repo-id local/door_final
    python record_scripted.py stacking --num-episodes 5 --dataset-dir ../data/stacking_mujoco_5ep
    python record_scripted.py pickplace --dry-run --num-episodes 3   # prueba sin grabar
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco
import torch
import yaml

import play_mujoco  # deja en sys.path los plugins locales, metrics/ y el site-packages de icubenv
from play_mujoco import _apply_control_arms, _discard_episode
import demo_door_mujoco as door
from demo_door_mujoco import Trajectory, mat2quat
from dependencies.door_scene import place_door, set_handle_side

repo_root = Path(__file__).resolve().parent
SCENES_DIR = repo_root / "dependencies" / "assets" / "scenes"
OTHER = {"right": "left", "left": "right"}


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def scene_entry(xml_name: str) -> dict:
    for e in yaml.safe_load((SCENES_DIR / "scenes.yaml").read_text(encoding="utf-8")) or []:
        if e["file"] == xml_name:
            return e
    raise SystemExit(f"{xml_name} no está en scenes.yaml")


def body_pos(m, d, name):
    return d.xpos[m.body(name).id].copy()


# ── Agarre de cubo de la mano derecha ────────────────────────────────────────
# Sacado de las 50 demos teleoperadas de lifting (local_final_mujoco), en el instante en que
# se cierra la mano: dedos hacia adelante y algo abajo, palma hacia abajo y hacia la izquierda,
# y el sitio de la mano 2.6 cm atrás, 2.5 cm a la derecha y 3.7 cm sobre el centro del cubo.
_CUBE_FINGERS = unit([0.94, 0.13, -0.25])
_CUBE_PALM = np.array([-0.27, 0.53, -0.78])
_CUBE_Z = unit(_CUBE_PALM - np.dot(_CUBE_PALM, _CUBE_FINGERS) * _CUBE_FINGERS)
R_CUBE = np.column_stack([_CUBE_FINGERS, np.cross(_CUBE_Z, _CUBE_FINGERS), _CUBE_Z])
CUBE_OFFSET = np.array([-0.026, -0.025, 0.037])


# Agarre de costado (como un vaso) para objetos verticales: palma hacia la izquierda, dedos
# hacia adelante y algo arriba, pulgar arriba; los dedos envuelven un eje vertical. Con el de
# cubo envuelven uno horizontal y una clavija de pie se acuesta en la mano.
_SIDE_FINGERS = unit([0.95, 0.0, 0.3])
_SIDE_PALM = unit([0.0, 1.0, 0.0])
R_SIDE = np.column_stack([_SIDE_FINGERS, np.cross(_SIDE_PALM, _SIDE_FINGERS), _SIDE_PALM])


def pick_and_place_plan(cube, place_center, rng, carry_z=0.12, release_dz=0.0, offset=CUBE_OFFSET,
                        approach_z=0.08, start=None, R=R_CUBE):
    """Plan de la mano derecha: tomar el cubo en `cube` y soltarlo con su centro en
    `place_center` (+ release_dz). Cada tramo: (duración, pos, R, mano, etiqueta).
    start = (posición, R) de la mano: si se da, primero sube approach_z desde ahí sin girar
    (objetos altos, que la mano botaría yendo en diagonal desde home)."""
    grasp = cube + offset
    place = place_center + np.array([0.0, 0.0, release_dz]) + offset
    up = np.array([0.0, 0.0, 1.0])
    j = lambda: rng.uniform(0.9, 1.1)  # noqa: E731  (velocidad variable entre episodios)
    lift = [] if start is None else [(1.5, np.asarray(start[0]) + approach_z * up, start[1], 0.0, "sube la mano")]
    return lift + [
        (2.5 * j(), grasp + approach_z * up, R, 0.0, "sobre el cubo"),
        (1.5 * j(), grasp, R, 0.0, "baja al cubo"),
        # El IK va ~4 cm atrás del target: sin esta espera la mano se cierra bajando y empuja
        # el cubo contra la mesa
        (1.0, grasp, R, 0.0, "espera"),
        (3.5, grasp, R, 1.0, "cierra la mano"),
        (2.0 * j(), grasp + carry_z * up, R, 1.0, "levanta"),
        (3.0 * j(), place + carry_z * up, R, 1.0, "lleva"),
        (1.0, place + carry_z * up, R, 1.0, "espera arriba"),   # que llegue antes de bajar
        (2.0 * j(), place, R, 1.0, "baja"),
        (0.8, place, R, 1.0, "espera abajo"),
        (2.5, place, R, 0.0, "suelta"),
        (1.5 * j(), place + 0.08 * up, R, 0.0, "se retira"),
    ]


def shift_from(traj, label, delta, max_shift=0.05):
    """Corrige en xy los waypoints desde el tramo `label`. Más de max_shift = el objeto ya no
    está en la mano (se cayó): no se corrige."""
    delta = np.array([delta[0], delta[1], 0.0])
    if np.linalg.norm(delta) <= max_shift:
        traj.shift(next(i for i, wp in enumerate(traj.plan) if wp[4] == label), delta)


# ── Tareas ───────────────────────────────────────────────────────────────────
class Task:
    """scene: XML; arms: brazos grabados (control_arms); plan() devuelve {brazo: waypoints}.
    grip / cube_offset: cierre de la mano y agarre por defecto de la tarea (los cambian --grip
    y --cube-offset)."""
    scene = ""
    arms = "right"
    grip = None
    cube_offset = CUBE_OFFSET

    def __init__(self, args):
        self.args = args

    def prepare(self, m, d, tc, rng) -> dict:
        """Cambios de la escena antes del reset de objetos (modelo). Devuelve info del episodio."""
        return {}

    def plan(self, m, d, tc, rng) -> tuple[dict, np.ndarray, dict]:
        raise NotImplementedError

    def success(self, m, d) -> bool:
        raise NotImplementedError

    def metrics(self, m, d) -> dict:
        return {}

    def correct(self, m, d, trajs, label) -> None:
        """Corrección en lazo cerrado al empezar el tramo `label` (por defecto, ninguna)."""


class DoorTask(Task):
    """Puerta con pestillo (escenario final: manilla a la izquierda, mano derecha)."""
    scene = "scene_icub_door.xml"

    def __init__(self, args):
        super().__init__(args)
        self.sides = ["right", "left"] if args.handle_sides == "both" else [args.handle_sides]
        arm_for = {"same": lambda s: s, "opposite": lambda s: OTHER[s],
                   "right": lambda s: "right", "left": lambda s: "left"}[args.arm]
        self.arm_for = {s: arm_for(s) for s in self.sides if s != "xml"}
        used = set(self.arm_for.values()) or {args.arm if args.arm in ("right", "left") else "right"}
        self.arms = used.pop() if len(used) == 1 else "both"
        self.count = {s: 0 for s in self.sides}

    def prepare(self, m, d, tc, rng):
        # Lado con menos episodios guardados (alterna); "xml" = el del XML (izquierda)
        side = min(self.sides, key=lambda s: (self.count[s], rng.random()))
        if side != "xml":
            set_handle_side(m, side)
        mujoco.mj_forward(m, d)
        place_door(m, d, (rng.uniform(*self.args.door_x), rng.uniform(*self.args.door_y)))
        self.side = side
        return {"handle_side": side}

    def plan(self, m, d, tc, rng):
        from dependencies.door_scene import handle_side
        side = handle_side(m)
        arm = self.arm_for.get(self.side, "right" if self.args.arm in ("opposite", "right") and side == "left"
                               else "left" if self.args.arm in ("opposite", "left") else side)
        pa = door.fill_plan_defaults(SimpleNamespace(
            grasp="top", back=None, up=None, twist=1.0, close_s=3.5,
            press_deg=rng.uniform(35.0, 45.0), open_deg=rng.uniform(50.0, 60.0),
            pull_s=rng.uniform(6.0, 8.0), press_s=rng.uniform(1.2, 1.8),
            grip_shift=rng.uniform(*self.args.grip_shift)))
        pa.back += rng.uniform(-0.005, 0.005)
        pa.up += rng.uniform(-0.005, 0.005)
        plan, bar = door.build_plan(m, d, pa, arm=arm)
        plans = {arm: plan}
        if self.arms == "both":
            plans[OTHER[arm]] = door.idle_plan(m, d, OTHER[arm])
        info = {"arm": arm, "panel_xy": [round(float(v), 4) for v in body_pos(m, d, "door_panel")[:2]],
                "press_deg": round(pa.press_deg, 1), "grip_shift": round(pa.grip_shift, 4)}
        return plans, bar, info

    def success(self, m, d):
        ok = door.door_open(m, d)
        if ok:
            self.count[self.side] += 1
        return ok

    def metrics(self, m, d):
        return {"door_deg": round(float(np.degrees(d.qpos[m.jnt_qposadr[m.joint("door_hinge").id]])), 1),
                "handle_deg": round(float(np.degrees(d.qpos[m.jnt_qposadr[m.joint("handle_hinge").id]])), 1)}


class StackingTask(Task):
    """Cubo azul encima del rojo, con la mano derecha."""
    scene = "scene_icub_stack.xml"
    # Ajustados en MuJoCo: con el agarre de lifting (0.037 y cierre 0.5) el cubo se suelta
    grip = 0.8
    cube_offset = np.array([-0.026, -0.025, 0.037])

    def plan(self, m, d, tc, rng):
        blue, red = body_pos(m, d, "blue-cube"), body_pos(m, d, "red-cube")
        # Centro del azul 5 cm sobre el del rojo (+1.5 cm de holgura al soltar: los dedos
        # sobresalen bajo el cubo y, más abajo, botan el rojo)
        plan = pick_and_place_plan(blue, red + np.array([0.0, 0.0, 0.05]), rng, release_dz=0.015,
                                   offset=self.args.cube_offset)
        return {"right": plan}, (blue + red) / 2, {"blue": blue.round(3).tolist(), "red": red.round(3).tolist()}

    def correct(self, m, d, trajs, label):
        # Con el cubo en la mano sobre el rojo: alinea en xy el cubo (no la mano) con el rojo,
        # como un operador que mira antes de bajar
        if label == "baja":
            shift_from(trajs["right"], "baja", body_pos(m, d, "red-cube") - body_pos(m, d, "blue-cube"))

    def success(self, m, d):
        blue, red = body_pos(m, d, "blue-cube"), body_pos(m, d, "red-cube")
        return bool(np.linalg.norm((blue - red)[:2]) < 0.025 and abs(blue[2] - red[2] - 0.05) < 0.01)

    def metrics(self, m, d):
        blue, red = body_pos(m, d, "blue-cube"), body_pos(m, d, "red-cube")
        return {"blue_minus_red": (blue - red).round(3).tolist(), "red_z": round(float(red[2]), 3)}


class PickPlaceTask(Task):
    """Cubo rojo dentro de la caja verde, con la mano derecha."""
    scene = "scene_icub_pickplace.xml"
    # Cubo de 4 cm (no 5): agarre más bajo y más cerrado que en lifting
    grip = 0.8
    cube_offset = np.array([-0.026, -0.025, 0.028])

    def plan(self, m, d, tc, rng):
        cube, bin_ = body_pos(m, d, "pick-cube"), body_pos(m, d, "place-bin")
        # Suelta con el cubo 2 cm sobre el borde de las paredes (techo de la caja: 0.04 + 0.02)
        plan = pick_and_place_plan(cube, bin_ + np.array([0.0, 0.0, 0.0225 + 0.02]), rng, release_dz=0.04,
                                   offset=self.args.cube_offset)
        return {"right": plan}, (cube + bin_) / 2, {"cube": cube.round(3).tolist(), "bin": bin_.round(3).tolist()}

    def success(self, m, d):
        rel = body_pos(m, d, "pick-cube") - body_pos(m, d, "place-bin")
        return bool(abs(rel[0]) < 0.055 and abs(rel[1]) < 0.055 and rel[2] < 0.045)

    def metrics(self, m, d):
        return {"cube_minus_bin": (body_pos(m, d, "pick-cube") - body_pos(m, d, "place-bin")).round(3).tolist()}


def mirror_grasp(R_right):
    """Orientación de la mano izquierda espejo (en y) de una de la derecha. La palma es +z local
    en la derecha y -z local en la izquierda."""
    M = np.diag([1.0, -1.0, 1.0])
    x_l, z_l = M @ R_right[:, 0], -(M @ R_right[:, 2])
    return np.column_stack([x_l, np.cross(z_l, x_l), z_l])


class TwoArmLiftTask(Task):
    """Olla tomada con ambas manos y levantada."""
    scene = "scene_icub_twoarmlift.xml"
    arms = "both"
    grip = 0.8

    def plan(self, m, d, tc, rng):
        pot = m.body("lift-pot").id
        self.z0 = float(d.xpos[pot][2])
        up = np.array([0.0, 0.0, 1.0])
        plans = {}
        for arm, sign in (("right", -1.0), ("left", 1.0)):
            # Tramo lateral del asa más cercano al robot (va en y): con los dedos hacia
            # adelante, como en el agarre de cubo, queda cruzado bajo ellos y se engancha al
            # cerrar. La izquierda usa el espejo del agarre de la derecha. (Dedos hacia la olla
            # sobre la barra exterior no: el IK no logra esa orientación, menos sin torso.)
            # En el mundo: la olla es simétrica y aparece girada 180° (como en Gazebo), así que
            # sus ejes locales no dicen cuál tramo queda cerca del robot
            bar = d.xpos[pot] + np.array([-0.045, sign * 0.115, 0.05])
            R = R_CUBE if arm == "right" else mirror_grasp(R_CUBE)
            grasp = bar + np.array([-0.03, 0.0, 0.035])
            plans[arm] = [
                (3.0, grasp + 0.08 * up, R, 0.0, "sobre el asa"),
                (1.5, grasp, R, 0.0, "baja al asa"),
                (1.0, grasp, R, 0.0, "espera"),
                (3.5, grasp, R, 1.0, "cierra la mano"),
                (4.0 * rng.uniform(0.9, 1.1), grasp + 0.10 * up, R, 1.0, "levanta"),
                (2.0, grasp + 0.10 * up, R, 1.0, "sostiene"),
            ]
        return plans, d.xpos[pot].copy(), {"pot": d.xpos[pot].round(3).tolist()}

    def success(self, m, d):
        # Levantada al menos 5 cm y derecha (con una sola mano queda colgando torcida)
        met = self.metrics(m, d)
        return met["pot_dz"] > 0.05 and met["tilt_deg"] < 15.0

    def metrics(self, m, d):
        pot = m.body("lift-pot").id
        tilt = np.degrees(np.arccos(np.clip(d.xmat[pot].reshape(3, 3)[2, 2], -1, 1)))
        touch = {"r": 0, "l": 0}
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if pot in (b1, b2):
                name = m.body(b2 if b1 == pot else b1).name
                if name[:7] in ("r_hand_", "l_hand_"):
                    touch[name[0]] += 1
        return {"pot_dz": round(float(d.xpos[pot][2]) - getattr(self, "z0", 0.0), 3),
                "tilt_deg": round(float(tilt), 1), "touch": touch}


class PegInHoleTask(Task):
    """Two-Arm Peg-In-Hole (como TwoArmPegInHole de robosuite): la derecha levanta la tabla vertical
    por el mango de arriba y la acerca al centro; la izquierda toma la clavija de pie, la gira 90°
    sobre el eje de los dedos (palma abajo) y la pasa de lado por el agujero (en -y), en el aire.
    Plan sin ajustar."""
    scene = "scene_icub_peginhole.xml"
    arms = "both"
    grip = None
    # Agarre de costado de la izquierda (espejo de R_SIDE): la palma a la izquierda de la
    # clavija y algo atrás
    cube_offset = np.array([-0.03, 0.03, 0.0])
    HALF = 0.065          # media altura de la clavija
    # Se toma por el centro: más abajo el IK no llega al agarre de costado y más arriba, al girar
    # con la palma abajo (la palma arriba el IK no la logra), el tramo que queda hacia la tabla
    # (-y, la punta de arriba) es muy corto para cruzarla
    GRASP_DZ = 0.0
    BOARD_HALF = 0.015    # media hondura del agujero (tabla de 3 cm)
    BOARD_LIFT = 0.06     # la tabla se sostiene 6 cm sobre su pose en la mesa...
    BOARD_MEET_Y = -0.03  # ...y en y = -0.03: el IK izquierdo pierde orientación más a la derecha
    HANDLE_OFFSET = CUBE_OFFSET   # el mango (bloque de 5 cm arriba de la tabla) se toma como el cubo

    def _peg_ends(self, m, d):
        b = m.body("peg").id
        axis = d.xmat[b].reshape(3, 3)[:, 2]
        return d.xpos[b] - self.HALF * axis, d.xpos[b] + self.HALF * axis

    def plan(self, m, d, tc, rng):
        board, peg = m.body("hole-board").id, body_pos(m, d, "peg")
        self.z0 = float(d.xpos[board][2])
        up = np.array([0.0, 0.0, 1.0])
        left_dir = np.array([0.0, 1.0, 0.0])
        j = lambda: rng.uniform(0.9, 1.1)  # noqa: E731

        # Derecha: el mango de arriba se toma como un cubo; la tabla se levanta y se lleva a
        # BOARD_MEET_Y
        handle = d.xpos[board] + d.xmat[board].reshape(3, 3) @ np.array([0.0, 0.0, 0.085])
        grasp = handle + self.HANDLE_OFFSET
        meet = np.array([0.0, self.BOARD_MEET_Y - d.xpos[board][1], self.BOARD_LIFT])
        right = [
            (3.0 * j(), grasp + 0.08 * up, R_CUBE, 0.0, "sobre el mango"),
            (1.5, grasp, R_CUBE, 0.0, "baja al mango"),
            (1.0, grasp, R_CUBE, 0.0, "espera"),
            (3.5, grasp, R_CUBE, 1.0, "cierra la mano"),
            (2.0 * j(), grasp + self.BOARD_LIFT * up, R_CUBE, 1.0, "levanta la tabla"),
            (2.0 * j(), grasp + meet, R_CUBE, 1.0, "lleva la tabla"),
            (14.0, grasp + meet, R_CUBE, 1.0, "sostiene"),
        ]

        # Izquierda: toma la clavija de costado, la levanta, la gira 90° en x (casi el eje de los
        # dedos; palma abajo) mientras la lleva al lado del agujero, con la punta 3 cm a la izquierda de la
        # cara de la tabla, y la empuja en -y hasta que su centro quede 3.5 cm a la izquierda del
        # agujero: los dedos (~2 cm a cada lado del agarre) quedan justo antes de la cara y la
        # punta sale 1.5 cm por la otra. La mano se ubica para que el centro de la clavija (fijo en la mano) caiga ahí.
        R = mirror_grasp(R_SIDE)
        R_turn = door.rot_axis([1.0, 0.0, 0.0], np.pi / 2) @ R   # z del mundo (clavija) -> -y
        off = self.args.cube_offset
        site = d.site_xpos[tc.lh_site_id].copy()
        g = peg + np.array([0.0, 0.0, self.GRASP_DZ]) + off
        center_in_hand = R.T @ (peg - g)
        hand_for = lambda center: center - R_turn @ center_in_hand  # noqa: E731
        hole_up = d.site_xpos[m.site("hole_center").id] + meet
        aside = hand_for(hole_up + (self.BOARD_HALF + 0.03 + self.HALF) * left_dir)
        inside = hand_for(hole_up + (self.BOARD_HALF + 0.02) * left_dir)
        lift_z = max(aside[2], g[2] + 0.10)
        left = [
            (1.5, site + 0.12 * up, d.site_xmat[tc.lh_site_id].reshape(3, 3).copy(), 0.0, "sube la mano"),
            (2.5 * j(), g + 0.12 * up, R, 0.0, "sobre la clavija"),
            (1.5 * j(), g, R, 0.0, "baja a la clavija"),
            (1.0, g, R, 0.0, "espera"),
            (3.5, g, R, 1.0, "cierra la mano"),
            (2.5 * j(), np.array([g[0], g[1], lift_z]), R, 1.0, "levanta"),
            (3.5 * j(), aside, R_turn, 1.0, "gira y lleva"),
            (1.5, aside, R_turn, 1.0, "espera al lado"),
            (2.5 * j(), inside, R_turn, 1.0, "empuja"),
            (2.0, inside, R_turn, 1.0, "espera adentro"),
        ]
        return ({"right": right, "left": left}, (peg + d.xpos[board]) / 2,
                {"board": d.xpos[board].round(3).tolist(), "peg": peg.round(3).tolist()})

    def correct(self, m, d, trajs, label, max_shift=0.05):
        # Al lado de la tabla y al empujar: alinea en x y z el eje de la clavija con el agujero
        # (shift_from corrige solo xy; acá se empuja en y)
        if label in ("espera al lado", "empuja"):
            delta = d.site_xpos[m.site("hole_center").id] - body_pos(m, d, "peg")
            delta[1] = 0.0
            if np.linalg.norm(delta) <= max_shift:
                traj = trajs["left"]
                traj.shift(next(i for i, wp in enumerate(traj.plan) if wp[4] == label), delta)

    def success(self, m, d):
        met = self.metrics(m, d)
        x, _, z = met["peg_in_board"]
        # Tabla en el aire, clavija cruzando la tabla (un extremo a cada lado), por dentro del
        # agujero (6 x 6 cm, holgura de 1 cm) y casi paralela a su eje
        return (met["board_dz"] > 0.03 and met["through"] and abs(x) < 0.02 and abs(z) < 0.02
                and met["angle_deg"] < 20.0)

    def metrics(self, m, d):
        board = m.body("hole-board").id
        Rb = d.xmat[board].reshape(3, 3)
        c = d.site_xpos[m.site("hole_center").id]
        a, b = (Rb.T @ (p - c) for p in self._peg_ends(m, d))
        # Punto donde el eje de la clavija corta el plano de la tabla (y = 0 en su marco)
        t = a[1] / (a[1] - b[1]) if abs(a[1] - b[1]) > 1e-6 else 0.0
        cross = a + t * (b - a)
        axis = unit(b - a)
        touch = {"r": 0, "l": 0}
        peg = m.body("peg").id
        for con in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            for obj, other in ((board, b2), (peg, b2), (board, b1), (peg, b1)):
                if obj in (b1, b2) and other != obj:
                    name = m.body(other).name
                    if name[:7] in ("r_hand_", "l_hand_"):
                        touch[name[0]] += 1
        return {"board_dz": round(float(d.xpos[board][2]) - getattr(self, "z0", 0.0), 3),
                "board_tilt_deg": round(float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))), 1),
                "peg_in_board": cross.round(3).tolist(),
                "through": bool(min(a[1], b[1]) < -self.BOARD_HALF and max(a[1], b[1]) > self.BOARD_HALF),
                "angle_deg": round(float(np.degrees(np.arccos(np.clip(abs(axis[1]), -1, 1)))), 1),
                "touch": touch}


TASKS = {"door": DoorTask, "stacking": StackingTask, "pickplace": PickPlaceTask,
         "twoarmlift": TwoArmLiftTask, "peginhole": PegInHoleTask}


# ── Grabación ────────────────────────────────────────────────────────────────
def reset_episode(robot, task, rng, settle_s=1.0) -> dict:
    """Robot en home, manos abiertas, escena de la tarea y targets sobre las manos."""
    tc = robot.teleop_core
    m, d = tc.model, tc.data
    mujoco.mj_resetData(m, d)
    info = task.prepare(m, d, tc, rng)
    d.qpos[:] = tc.home_qpos            # qpos0 del XML: robot y objetos en su pose del XML
    d.ctrl[:] = tc.home_ctrl
    tc._apply_home_pose_to_arrays(d.qpos, d.ctrl)
    tc._hand_goal = {"right": None, "left": None}
    tc._hand_last_t = None
    tc._set_hand_level("right", 0.0)
    tc._set_hand_level("left", 0.0)
    tc._reset_scenario()                # objetos (rangos de scenes.yaml) y joints de la escena
    mujoco.mj_forward(m, d)
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    tc._snap_targets_to_ee()
    return info


def scripted_action(robot, trajs, t, gaze_world):
    """Acción con las mismas claves y frame que la del teleoperador (iCubTeleop.get_action)."""
    a = {}
    for arm, traj in trajs.items():
        side = arm[0] + "h"
        if f"{side}_pos_x" not in robot.action_features:
            continue
        pos, quat, hand = traj.at(t)[:3]
        a.update({f"{side}_pos_{ax}": v for ax, v in zip("xyz", pos)})
        a.update({f"{side}_quat_{ax}": v for ax, v in zip("wxyz", quat)})
        a[f"{side}_gripper"] = hand
    if robot.config.use_gaze:
        a.update({f"gaze_{ax}": v for ax, v in zip("xyz", gaze_world)})
    if robot.config.pose_frame == "root":
        robot._convert_poses(a, robot._ACTION_POSE_GROUPS, to_root=True)
    return {k: torch.tensor(float(v), dtype=torch.float32) for k, v in a.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=list(TASKS))
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--repo-id", default=None, help="Por defecto local/<tarea>_mujoco_<N>ep")
    p.add_argument("--root", default=str(repo_root.parent / "data"),
                   help="Carpeta base; el dataset queda en <root>/<repo_id>_<fecha> como en el Hub")
    p.add_argument("--dataset-dir", default=None, help="Carpeta exacta del dataset (en vez de --root)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--config", default=str(repo_root / "utils" / "control_config.yaml"))
    p.add_argument("--max-attempts", type=int, default=None, help="Por defecto 3 x num-episodes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Sin dataset ni cámaras: solo prueba el plan")
    p.add_argument("--trace", action="store_true", help="Imprime el estado al terminar cada tramo del plan")
    p.add_argument("--grip", type=float, default=None,
                   help="Fracción de cierre de la mano con gripper=1 (por defecto hand_close_ratio del YAML)")
    p.add_argument("--cube-offset", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"),
                   help="Sitio de la mano respecto del centro del cubo al agarrar (mundo) [m] "
                        "(por defecto el de la tarea)")
    g = p.add_argument_group("door")
    g.add_argument("--handle-sides", choices=["xml", "right", "left", "both"], default="xml",
                   help="Lado de la manilla (xml = izquierda); both = mitad y mitad")
    g.add_argument("--arm", choices=["right", "left", "same", "opposite"], default="right",
                   help="Mano que abre: fija, la del lado de la manilla o la contraria")
    g.add_argument("--door-x", type=float, nargs=2, default=[0.45, 0.50], metavar=("MIN", "MAX"),
                   help="Rango del centro de la hoja en x [m] (XML: 0.47; más cerca que 0.45 la mano "
                        "derecha en home roza la puerta)")
    g.add_argument("--door-y", type=float, nargs=2, default=[-0.16, -0.10], metavar=("MIN", "MAX"),
                   help="Rango del centro de la hoja en y [m] (XML: -0.13)")
    g.add_argument("--grip-shift", type=float, nargs=2, default=[0.015, 0.03], metavar=("MIN", "MAX"),
                   help="Agarre corrido por la barra hacia su punta [m]")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    try:  # LeRobot reciente: movidas a lerobot.utils.feature_utils
        from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
    except ImportError:
        from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION, OBS_STR

    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot_robot_icub.icub_mujoco import canonicalize_quats
    from lerobot_teleoperator_icubteleop.config_icubteleop import iCubTeleopConfig

    task = TASKS[args.task](args)
    entry = scene_entry(task.scene)
    instruction = entry["task"]
    model_path = SCENES_DIR / task.scene
    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path), config_path=args.config, vr_enabled=False, vr_ip=None,
        scene_objects=entry.get("objects", []), scene_joints=entry.get("joints", []),
    )
    teleop_cfg = iCubTeleopConfig(model_path=str(model_path), config_path=args.config,
                                  vr_enabled=False, vr_ip=None)
    _apply_control_arms(robot_cfg, teleop_cfg, task.arms, Path(args.config))
    if entry.get("cameras"):
        robot_cfg.camera_names = list(entry["cameras"])
    if args.dry_run:
        robot_cfg.camera_names = []
    robot = make_robot_from_config(robot_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset = dataset_root = None
    if not args.dry_run:
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )
        repo_id = args.repo_id or f"local/{args.task}_mujoco_{args.num_episodes}ep"
        dataset_root = (Path(args.dataset_dir).expanduser() if args.dataset_dir else
                        Path(args.root).expanduser() / f"{repo_id.replace('/', '_')}_"
                                                       f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        dataset = LeRobotDataset.create(repo_id, args.fps, root=dataset_root, robot_type=robot.name,
                                        features=features, use_videos=True)
        print(f"[rec] dataset: {dataset_root} ({repo_id})")
    print(f"[rec] tarea {args.task}: {instruction!r} | brazos {task.arms} | cámaras {robot_cfg.camera_names}")

    robot.connect()
    tc = robot.teleop_core
    m, d = tc.model, tc.data
    grip = args.grip if args.grip is not None else task.grip
    if grip is not None:
        tc._hand_close_goal = tc._hand_open + float(np.clip(grip, 0, 1)) * (tc._hand_close_full - tc._hand_open)
    args.cube_offset = np.asarray(args.cube_offset if args.cube_offset is not None else task.cube_offset, dtype=float)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)          # _reset_scenario de la escena usa np.random
    max_attempts = args.max_attempts or 3 * args.num_episodes
    log, saved = [], 0

    try:
        attempt = 0
        while saved < args.num_episodes and attempt < max_attempts:
            attempt += 1
            info = reset_episode(robot, task, rng)
            plans, gaze, plan_info = task.plan(m, d, tc, rng)
            mocap = {"right": tc.rh_mocap_id, "left": tc.lh_mocap_id}
            trajs = {arm: Trajectory(plan, d.mocap_pos[mocap[arm]], d.mocap_quat[mocap[arm]])
                     for arm, plan in plans.items()}
            duration = max(tr.duration for tr in trajs.values())
            n_frames = int(np.ceil(duration * args.fps))
            t0, t_wall = d.time, time.perf_counter()
            seg = {arm: 0 for arm in trajs}
            labels = {arm: None for arm in trajs}
            for _ in range(n_frames):
                for arm, tr in trajs.items():
                    label = tr.at(d.time - t0)[3]
                    if label != labels[arm]:
                        task.correct(m, d, trajs, label)
                        labels[arm] = label
                if args.trace:
                    for arm, tr in trajs.items():
                        i = tr.at(d.time - t0)[4]
                        if i != seg[arm]:
                            site = tc.rh_site_id if arm == "right" else tc.lh_site_id
                            dur, pos, R, hand, label = tr.plan[seg[arm]]
                            print(f"   [{arm}] {label:<16} mano a {np.linalg.norm(d.site_xpos[site] - pos) * 100:4.1f} cm "
                                  f"{task.metrics(m, d)}", flush=True)
                            seg[arm] = i
                obs = None if args.dry_run else robot.get_observation()
                act = scripted_action(robot, trajs, d.time - t0, gaze)
                act_processed = teleop_action_processor((act, obs or {}))
                robot.send_action(robot_action_processor((act_processed, obs or {})))
                if dataset is not None:
                    recorded = act_processed
                    if robot.config.canonical_quats and robot.config.pose_frame == "root":
                        recorded = canonicalize_quats(dict(recorded))
                    dataset.add_frame({
                        **build_dataset_frame(dataset.features, robot_observation_processor(obs), prefix=OBS_STR),
                        **build_dataset_frame(dataset.features, recorded, prefix=ACTION),
                        "task": instruction,
                    })

            ok = task.success(m, d)
            entry_log = {"attempt": attempt, "saved": ok, "frames": n_frames, **info, **plan_info,
                         **task.metrics(m, d)}
            if ok:
                if dataset is not None:
                    entry_log["episode"] = dataset.num_episodes
                    dataset.save_episode()
                saved += 1
            elif dataset is not None:
                _discard_episode(dataset)
            log.append(entry_log)
            print(f"[rec] intento {attempt}: {task.metrics(m, d)} -> {'GUARDADO' if ok else 'descartado'} "
                  f"({saved}/{args.num_episodes}, {time.perf_counter() - t_wall:.0f} s)", flush=True)
    finally:
        if dataset is not None and dataset.num_episodes:
            dataset.finalize()
            # Registro de la generación (fuera de meta/ para no interferir con LeRobot)
            (dataset_root / "scripted_generation.json").write_text(json.dumps({
                "script": Path(__file__).name, "task": args.task, "scene": task.scene,
                "instruction": instruction, "seed": args.seed, "arms": task.arms,
                "cameras": list(robot_cfg.camera_names), "args": vars(args), "attempts": log,
            }, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)), encoding="utf-8")
        if robot.is_connected:
            robot.disconnect()

    print(f"\n[rec] {saved} episodios {'logrados' if args.dry_run else 'guardados'} de {len(log)} intentos"
          + (f" -> {dataset_root}" if dataset_root else ""))


if __name__ == "__main__":
    main()
