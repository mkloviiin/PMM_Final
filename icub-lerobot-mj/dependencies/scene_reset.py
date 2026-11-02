"""Reset aleatorio de las escenas de MuJoCo, según su entrada en scenes.yaml.

Un solo sorteo para todos los caminos: el reset del núcleo MuJoCo (botón Y del VR, tecla N,
inicio y fin de episodio en el Hub), eval_mujoco.py y record_scripted.py. El sorteo da un
`layout` (lado de la tarea, poses de los objetos, posición de la puerta y joints) que
apply_layout escribe en un modelo; el viewer del teleoperador aplica el mismo layout.

Claves de scenes.yaml que usa (además de objects/joints):
  sides: right | left | alternate
      Lado de la tarea. Los rangos y de `objects` son los del lado derecho (el objeto que se
      toma, a la derecha del robot: y < 0); "left" los espeja en y. En la puerta, el lado de
      la manilla. alternate = derecha, izquierda, derecha...: cada reset pasa al otro lado.
  door: {panel_xy: [x, y], jitter: m}
      Centro de la hoja de la puerta, más un uniforme de ±jitter en x y en y.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from dependencies.door_scene import handle_side, place_door, set_handle_side

SCENES_YAML = Path(__file__).resolve().parent / "assets" / "scenes" / "scenes.yaml"


def scene_entry(model_path) -> dict:
    """Entrada de scenes.yaml del XML `model_path` ({} si no está declarado)."""
    try:
        entries = yaml.safe_load(SCENES_YAML.read_text(encoding="utf-8")) or []
    except Exception:
        return {}
    name = Path(model_path).name
    return next((e for e in entries if e.get("file") == name), {})


def _sample(val, default=0.0):
    """Uniforme si es [min, max], fijo si es escalar."""
    if isinstance(val, list) and len(val) == 2:
        return float(np.random.uniform(val[0], val[1]))
    return float(val) if val is not None else float(default)


def _mirror_y(val):
    """Rango (o valor) y del lado derecho -> el del lado izquierdo."""
    if isinstance(val, list) and len(val) == 2:
        return [-val[1], -val[0]]
    return -float(val) if val is not None else None


def has_door(model) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge") >= 0


class SceneRandomizer:
    """Sorteo de layouts de una escena. Con sides: alternate cada sorteo pasa al otro lado
    (el primero es la derecha): N, Y, guardar o descartar en el Hub y cada episodio de eval."""

    def __init__(self, objects: list | None, joints: list | None, entry: dict | None = None):
        entry = entry or {}
        self.objects = list(objects or [])
        self.joints = list(joints or [])
        # Sin sides: los rangos tal cual y la puerta del lado del XML (side None)
        self.sides_mode = entry.get("sides")
        if self.sides_mode not in (None, "right", "left", "alternate"):
            raise ValueError(f"scenes.yaml sides: {self.sides_mode!r} (right, left o alternate)")
        self.door = entry.get("door")
        self.side = None if self.sides_mode is None else "left" if self.sides_mode == "left" else "right"
        self._sampled = False

    def sample(self, model, data, side: str | None = None, door_xy=None) -> dict:
        """Layout nuevo. side / door_xy fuerzan el lado o el centro de la hoja (overrides de
        eval_mujoco.py y record_scripted.py); un lado forzado no avanza la alternancia."""
        if side is None:
            if self._sampled and self.sides_mode == "alternate":
                self.side = "left" if self.side == "right" else "right"
            self._sampled = True
            side = self.side
        layout = {"side": side, "objects": {}, "joints": {}, "door_xy": None}
        for obj in self.objects:
            name, spawn = obj.get("body", ""), dict(obj.get("spawn", {}))
            try:
                body = model.body(name)
            except KeyError:
                print(f"[Reset] Skip '{name}': not in the model")
                continue
            jnt = body.jntadr[0]
            if jnt == -1:
                continue
            adr = model.jnt_qposadr[jnt]
            if side == "left" and "y" in spawn:
                spawn["y"] = _mirror_y(spawn["y"])
            pos = np.array([_sample(spawn.get(ax), data.qpos[adr + i]) for i, ax in enumerate("xyz")])
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
        body = model.body(name)
        jnt = body.jntadr[0]
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
        print(f"[Reset] {' | '.join(desc)}")
