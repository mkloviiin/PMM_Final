"""Reset aleatorio de las escenas de MuJoCo, según su entrada en scenes.yaml.

Un solo sorteo para todos los caminos: lo hace el núcleo MuJoCo del robot (MuJoCoTeleop.
_reset_scenario: botón Y del VR, tecla N, 'n' de la consola/Hub, inicio y fin de episodio en
play.py, cada episodio de eval_policy.py). El sorteo da un `layout` (lado de la tarea, poses de
los objetos, posición de la puerta y joints) que apply_layout escribe en un modelo; el viewer
del teleoperador aplica el mismo layout.

Claves de scenes.yaml que usa (además de objects/joints), por ahora solo en MuJoCo:
  sides: right | left | alternate
      Lado de la tarea. Los rangos y de `objects` son los del lado derecho (el objeto que se
      toma, a la derecha del robot: y < 0); "left" espeja en y todos los objetos. En la puerta,
      el lado de la manilla. alternate = derecha, izquierda, derecha...: cada reset pasa al otro.
      Sin sides, los objetos con spawn.mirror_y siguen con el sorteo 50/50 (como en Gazebo).
  door: {panel_xy: [x, y], jitter: m}
      Centro de la hoja de la puerta, más un uniforme de ±jitter en x y en y.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from tools.door_scene import handle_side, place_door, set_handle_side

SCENES_YAML = Path(__file__).resolve().parents[2] / "mujoco_sim" / "assets" / "scenes" / "scenes.yaml"


def scene_entry(model_path) -> dict:
    """Entrada de scenes.yaml del XML `model_path` ({} si no está declarado)."""
    try:
        entries = yaml.safe_load(SCENES_YAML.read_text(encoding="utf-8")) or []
    except Exception:
        return {}
    name = Path(model_path).name
    return next((e for e in entries if e.get("file") == name), {})


def _sample(val, default=0.0) -> float:
    """Uniforme si es [min, max], fijo si es escalar."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(np.random.uniform(val[0], val[1]))
    return float(val) if val is not None else float(default)


def has_door(model) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge") >= 0


class SceneRandomizer:
    """Sorteo de layouts de una escena. Con sides: alternate cada sorteo pasa al otro lado (el
    primero es la derecha); con objetos mirror_y y sin sides, 50/50 en cada sorteo."""

    def __init__(self, objects: list | None, joints: list | None, entry: dict | None = None):
        entry = entry or {}
        self.objects = list(objects or [])
        self.joints = list(joints or [])
        self.sides_mode = entry.get("sides")
        if self.sides_mode not in (None, "right", "left", "alternate"):
            raise ValueError(f"scenes.yaml sides: {self.sides_mode!r} (right, left o alternate)")
        # Objetos que se espejan: todos con sides; si no, los marcados con mirror_y
        self.mirrored = {o.get("body") for o in self.objects
                         if self.sides_mode is not None or o.get("spawn", {}).get("mirror_y")}
        if self.sides_mode is None and self.mirrored:
            self.sides_mode = "random"
        self.door = entry.get("door")
        self.side = "left" if self.sides_mode == "left" else "right" if self.sides_mode else None
        self._sampled = False

    def _next_side(self) -> str | None:
        if self.sides_mode == "random":
            return "left" if np.random.uniform() < 0.5 else "right"
        if self.sides_mode == "alternate" and self._sampled:
            self.side = "left" if self.side == "right" else "right"
        self._sampled = True
        return self.side

    def sample(self, model, data, side: str | None = None, door_xy=None) -> dict:
        """Layout nuevo. side / door_xy fuerzan el lado o el centro de la hoja (eval_policy.py);
        un lado forzado no avanza la alternancia."""
        side = side or self._next_side()
        layout = {"side": side, "objects": {}, "joints": {}, "door_xy": None}
        for obj in self.objects:
            name, spawn = obj.get("body", ""), obj.get("spawn", {})
            try:
                body = model.body(name)
            except KeyError:
                print(f"[Reset] Skip '{name}': not in the model")
                continue
            jnt = body.jntadr[0]
            if jnt == -1:
                continue
            adr = model.jnt_qposadr[jnt]
            pos = np.array([_sample(spawn.get(ax), data.qpos[adr + i]) for i, ax in enumerate("xyz")])
            if side == "left" and name in self.mirrored and "y" in spawn:
                pos[1] = -pos[1]
            quat = np.array(spawn.get("quat", [1, 0, 0, 0]), dtype=float)
            layout["objects"][name] = (pos, quat)
        for jnt in self.joints:
            layout["joints"][jnt.get("name", "")] = float(jnt.get("qpos", 0.0))
        if has_door(model):
            if door_xy is None and self.door:
                cx, cy = self.door["panel_xy"]
                j = float(self.door.get("jitter", 0.0))
                door_xy = (cx + np.random.uniform(-j, j), cy + np.random.uniform(-j, j))
            layout["door_xy"] = None if door_xy is None else tuple(float(v) for v in door_xy)
        return layout


def apply_layout(model, data, layout: dict, verbose: bool = True) -> None:
    """Escribe el layout en model/data: lado y posición de la puerta (modelo), joints de la
    escena y poses de los objetos (qpos, con velocidad cero)."""
    for name, qpos in layout["joints"].items():
        try:
            jid = model.joint(name).id
        except KeyError:
            continue
        data.qpos[model.jnt_qposadr[jid]] = qpos
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    if has_door(model):
        # Con la puerta cerrada (joints ya en 0): place_door mide el centro de la hoja en xpos
        if layout["side"] is not None:
            set_handle_side(model, layout["side"])
        mujoco.mj_forward(model, data)
        if layout["door_xy"] is not None:
            place_door(model, data, layout["door_xy"])
    for name, (pos, quat) in layout["objects"].items():
        jnt = model.body(name).jntadr[0]
        adr, dof = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = quat
        data.qvel[dof:dof + (6 if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE else 1)] = 0.0
    mujoco.mj_forward(model, data)
    if verbose:
        desc = [f"side={layout['side']}"]
        desc += [f"{n} → ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for n, (p, _) in layout["objects"].items()]
        if has_door(model):
            desc.append(f"handle={handle_side(model)}")
            if layout["door_xy"] is not None:
                desc.append(f"door panel → ({layout['door_xy'][0]:.3f}, {layout['door_xy'][1]:.3f})")
        print(f"[Reset] {' | '.join(desc)}", flush=True)
