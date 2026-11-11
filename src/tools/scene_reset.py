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
  objects[].variants: [{name, z, radius, mass}]  (Pick-and-Place, robosuite single_object_mode=1)
      Una variante al azar por reset; el cuerpo tiene la colisión y la malla de todas y
      set_variant deja activa una. z reemplaza spawn.z y radius achica los rangos x/y.
  objects[].spawn.yaw: [min, max] grados, giro uniforme en z (después de spawn.quat).
  fixtures: [{body, pos}]
      Cuerpos fijos en su posición del lado derecho; en el izquierdo se espeja y.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

from tools.door_scene import handle_side, place_door, set_handle_side

SCENES_YAML = Path(__file__).resolve().parents[2] / "mujoco_sim" / "assets" / "scenes" / "scenes.yaml"


def _model_objects(model) -> list:
    """Entradas `objects` de scenes.yaml de la escena cargada en `model` (por el nombre del modelo)."""
    try:
        entries = yaml.safe_load(SCENES_YAML.read_text(encoding="utf-8")) or []
    except Exception:
        return []
    model_name = bytes(model.names).split(b"\x00", 1)[0].decode()  # <mujoco model="..."> = nombre del XML
    for e in entries:
        if Path(e.get("file", "")).stem == model_name:
            return e.get("objects") or []
    return []


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


def _yaw_quat(quat, yaw_deg: float) -> np.ndarray:
    """quat (w, x, y, z) seguido de un giro de yaw_deg en z (marco del cuerpo)."""
    h = np.radians(yaw_deg) / 2
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(quat, dtype=float), np.array([np.cos(h), 0.0, 0.0, np.sin(h)]))
    return out


def fixture_positions(entry: dict, side: str | None) -> dict[str, np.ndarray]:
    """Posición (marco MuJoCo) de los fixtures de la escena en ese lado ("left" espeja y)."""
    out = {}
    for fx in entry.get("fixtures") or []:
        pos = np.array(fx["pos"], dtype=float)
        if side == "left":
            pos[1] = -pos[1]
        out[fx["body"]] = pos
    return out


def variant_names(obj: dict) -> list[str]:
    return [v["name"] for v in obj.get("variants") or []]


def set_variant(model, body: str, variants: list[dict], name: str) -> None:
    """Deja activa la variante `name` del cuerpo: su colisión y su malla (las demás sin contactos y
    en el grupo 5, que no se dibuja), y su masa e inercia (primitiva de colisión uniforme)."""
    for v in variants:
        on = v["name"] == name
        col = model.geom(f"{body}-{v['name']}-col").id
        vis = model.geom(f"{body}-{v['name']}-vis").id
        model.geom_contype[col] = model.geom_conaffinity[col] = int(on)
        model.geom_group[vis] = 1 if on else 5
        if not on:
            continue
        m = float(v["mass"])
        size = model.geom_size[col]
        if model.geom_type[col] == mujoco.mjtGeom.mjGEOM_BOX:
            a, b, c = 2 * size
            inertia = m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
        else:  # cilindro: radio, media altura
            r, h = size[0], 2 * size[1]
            inertia = np.array([m * (3 * r * r + h * h) / 12] * 2 + [m * r * r / 2])
        bid = model.body(body).id
        model.body_mass[bid] = m
        model.body_inertia[bid] = inertia
        model.body_ipos[bid] = model.geom_pos[col]
        model.body_iquat[bid] = [1, 0, 0, 0]


def init_variants(model) -> None:
    """Al cargar un modelo: primera variante de cada objeto con `variants` (el XML compila todas
    las colisiones activas, ver src/tools/pickplace_assets.py)."""
    for o in _model_objects(model):
        if o.get("variants"):
            try:
                set_variant(model, o["body"], o["variants"], o["variants"][0]["name"])
            except KeyError:
                pass


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
        self.entry = entry
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
        layout = {"side": side, "objects": {}, "joints": {}, "door_xy": None, "variants": {},
                  "fixtures": fixture_positions(self.entry, side)}
        for obj in self.objects:
            name, spawn = obj.get("body", ""), dict(obj.get("spawn", {}))
            variant = None
            if obj.get("variants"):
                # robosuite single_object_mode=1: una al azar; z propia y rangos achicados en su radio
                variant = obj["variants"][np.random.randint(len(obj["variants"]))]
                r = float(variant.get("radius", 0.0))
                spawn["z"] = variant["z"]
                for ax in "xy":
                    if isinstance(spawn.get(ax), (list, tuple)):
                        lo, hi = spawn[ax]
                        spawn[ax] = [lo + r, hi - r] if hi - lo > 2 * r else [(lo + hi) / 2] * 2
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
            if spawn.get("yaw") is not None:
                quat = _yaw_quat(quat, _sample(spawn["yaw"]))
            layout["objects"][name] = (pos, quat)
            if variant is not None:
                layout["variants"][name] = variant["name"]
        for jnt in self.joints:
            layout["joints"][jnt.get("name", "")] = float(jnt.get("qpos", 0.0))
        if has_door(model):
            if door_xy is None and self.door:
                cx, cy = self.door["panel_xy"]
                j = float(self.door.get("jitter", 0.0))
                door_xy = (cx + np.random.uniform(-j, j), cy + np.random.uniform(-j, j))
            layout["door_xy"] = None if door_xy is None else tuple(float(v) for v in door_xy)
        return layout


def apply_layout(model, data, layout: dict, verbose: bool = True, objects: list | None = None) -> None:
    """Escribe el layout en model/data: lado y posición de la puerta (modelo), fixtures (modelo),
    variante de cada objeto (modelo; `objects` = entradas de scenes.yaml con sus variants), joints
    de la escena y poses de los objetos (qpos, con velocidad cero)."""
    for name, pos in (layout.get("fixtures") or {}).items():
        try:
            model.body_pos[model.body(name).id] = pos
        except KeyError:
            continue
    variant_defs = {o.get("body"): o.get("variants") for o in objects or [] if o.get("variants")}
    for body, vname in (layout.get("variants") or {}).items():
        defs = variant_defs.get(body)
        if defs is None:
            # sin las entradas (p. ej. el viewer del teleoperador): las de scenes.yaml del modelo
            defs = next((o["variants"] for o in _model_objects(model) if o.get("body") == body), None)
        if defs:
            set_variant(model, body, defs, vname)
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
        desc += [f"{b}={v}" for b, v in (layout.get("variants") or {}).items()]
        desc += [f"{n} → ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" for n, (p, _) in layout["objects"].items()]
        if has_door(model):
            desc.append(f"handle={handle_side(model)}")
            if layout["door_xy"] is not None:
                desc.append(f"door panel → ({layout['door_xy'][0]:.3f}, {layout['door_xy'][1]:.3f})")
        print(f"[Reset] {' | '.join(desc)}", flush=True)
