#!/usr/bin/env python3
"""Genera gazebo/models/<body>/ desde un cuerpo de una escena de MuJoCo (objetos de las tareas).

Copia las cajas y cilindros del cuerpo (pose en el cuerpo, tamaño, color y fricción), su masa e
inercia y, si el objeto se mueve, el PosePublisher con que play_gazebo graba obj{i}_* (ver
gazebo/models/blue-cube). La geometría va tal cual, sin girar: Gazebo crea los objetos sin
rotación en su mundo, que está girado 180° respecto del de MuJoCo, y por eso en MuJoCo estos
objetos aparecen con quat (0, 0, 0, 1) (scenes.yaml): así se ven igual y graban la misma
orientación en el frame root.

Uso (entorno con mujoco, p. ej. icubenv), desde icub-lerobot-gazebo/:
    python tools/object_sdf_from_mujoco.py scene_icub_twoarmlift.xml lift-pot
    python tools/object_sdf_from_mujoco.py scene_icub_peginhole.xml hole-board
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

GZ_ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = GZ_ROOT.parent / "icub-lerobot-mj" / "dependencies" / "assets" / "scenes"
MODELS_DIR = GZ_ROOT / "gazebo" / "models"
# Color (como se ve en MuJoCo) de los materiales con textura (Gazebo no las carga)
MATERIAL_RGBA = {"plate_mat": (0.55, 0.27, 0.15)}


def linear(c: float) -> float:
    """sRGB -> lineal. MuJoCo muestra los colores tal cual y Gazebo (Ogre2) les aplica gamma:
    el color lineal en Gazebo se ve como el sRGB en MuJoCo."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def gz_color(rgb) -> str:
    return " ".join(f"{linear(float(c)):.3g}" for c in rgb) + " 1"


def rpy(R) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), la convención de <pose> en SDF."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])


def pose(p, q) -> str:
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, dtype=float))
    return " ".join(f"{v:.6g}" for v in (*p, *rpy(M.reshape(3, 3))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="XML de icub-lerobot-mj/dependencies/assets/scenes")
    ap.add_argument("body")
    ap.add_argument("--static", action="store_true", help="Pieza fija (fixture): sin inercia ni PosePublisher")
    ap.add_argument("--description", default=None)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENES_DIR / args.scene))
    bid = m.body(args.body).id
    collisions, visuals = [], []
    for gid in (g for g in range(m.ngeom) if m.geom_bodyid[g] == bid):
        g = m.geom(gid)
        if g.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
            shape = f"<box><size>{' '.join(f'{2 * v:.6g}' for v in g.size)}</size></box>"
        elif g.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = f"<cylinder><radius>{g.size[0]:.6g}</radius><length>{2 * g.size[1]:.6g}</length></cylinder>"
        else:
            raise SystemExit(f"Tipo de geom no soportado en {args.body}: {g.name or gid}")
        p = pose(m.geom_pos[gid], m.geom_quat[gid])
        name = g.name or f"geom{gid}"
        if m.geom_contype[gid] or m.geom_conaffinity[gid]:
            mu = float(m.geom_friction[gid][0])
            collisions.append(f"""      <collision name="{name}">
        <pose>{p}</pose>
        <geometry>{shape}</geometry>
        <surface><friction><ode><mu>{mu:.6g}</mu><mu2>{mu:.6g}</mu2></ode></friction></surface>
      </collision>""")
        mat = m.geom_matid[gid]
        visible = m.geom_rgba[gid][3] > 0
        color = gz_color(MATERIAL_RGBA.get(m.material(mat).name, (0.6, 0.6, 0.6)) if mat >= 0
                         else m.geom_rgba[gid][:3])
        if visible:
            visuals.append(f"""      <visual name="{name}_visual">
        <pose>{p}</pose>
        <geometry>{shape}</geometry>
        <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
      </visual>""")

    inertial = ""
    if not args.static:
        I = m.body_inertia[bid]
        inertial = f"""      <inertial>
        <pose>{pose(m.body_ipos[bid], m.body_iquat[bid])}</pose>
        <mass>{m.body_mass[bid]:.6g}</mass>
        <inertia><ixx>{I[0]:.6g}</ixx><iyy>{I[1]:.6g}</iyy><izz>{I[2]:.6g}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
"""
    pose_pub = "" if args.static else """
    <!-- Pose del modelo para play_gazebo (obj{i}_*, frame root) -->
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose>
      <publish_model_pose>true</publish_model_pose>
      <use_pose_vector_msg>false</use_pose_vector_msg>
      <update_frequency>30</update_frequency>
    </plugin>"""
    sdf = f"""<?xml version="1.0" ?>
<!-- GENERADO por tools/object_sdf_from_mujoco.py desde {args.scene} ({args.body}): no editar a mano. -->
<sdf version="1.7">
  <model name="{args.body}">
    <static>{"true" if args.static else "false"}</static>
    <link name="link">
{inertial}{chr(10).join(collisions)}
{chr(10).join(visuals)}
    </link>{pose_pub}
  </model>
</sdf>
"""
    out = MODELS_DIR / args.body
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.sdf").write_text(sdf, encoding="utf-8")
    desc = args.description or f"{args.body} de {args.scene} (generado desde MuJoCo)"
    (out / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{args.body}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>{desc}</description>
</model>
""", encoding="utf-8")
    print(out / "model.sdf")


if __name__ == "__main__":
    main()
