#!/usr/bin/env python3
"""Genera gazebo_sim/models/door/model.sdf desde la puerta de MuJoCo (scene_icub_door.xml).

Mismas medidas, masas, bisagras (door_hinge, handle_hinge con su resorte) y pestillo que en
MuJoCo, pasadas al mundo de Gazebo: (x, y, z)_gz = (-x, -y, z)_mj y +180° de yaw. Como
reset_objects.sh crea los modelos sin orientación, ese giro va dentro del modelo; su origen es
door_mount y la pose del fixture (scene_fixtures.yaml) es solo la posición.

Colisiones (collide_bitmask; el robot choca con todo):
  1 = hoja y postes del marco, 2 = pestillo (latch_tip) y sus trabas, 4 = manilla.
Así el pestillo solo choca con la traba (como contype/conaffinity 2 en MuJoCo) y la manilla,
que atraviesa la hoja en su eje, no choca con ella.

Uso (entorno Pixi del proyecto), desde la raíz del repo:
    pixi run python src/tools/door_sdf_from_mujoco.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO_ROOT / "mujoco_sim" / "assets" / "scenes" / "scene_icub_door.xml"
DEFAULT_OUT = REPO_ROOT / "gazebo_sim" / "models" / "door"

RZ180 = np.diag([-1.0, -1.0, 1.0])
LINKS = {"door_frame": "frame", "door_panel": "panel", "latch": "latch"}
BITMASK = {"r_frame": 1, "l_frame": 1, "panel_collision": 1, "latch_strike": 2, "latch_strike_mirror": 2,
           "latch_tip": 2, "handle_base": 4, "handle_collision": 4}
# Madera oscura y latón (texturas en MuJoCo) en lineal: Gazebo (Ogre2) les aplica gamma
_lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
WOOD, METAL = (" ".join(f"{_lin(c):.3g}" for c in rgb) + " 1" for rgb in ((0.36, 0.22, 0.12), (0.75, 0.6, 0.25)))
# Fricción de la manilla como la de los cubos de Gazebo (los dedos la tienen que agarrar)
MU = {"handle_collision": 5.0, "handle_base": 5.0}


def rpy(R) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), la convención de <pose> en SDF."""
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])


def quat_mat(q) -> np.ndarray:
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, dtype=float))
    return M.reshape(3, 3)


def pose(p, R) -> str:
    return " ".join(f"{v:.6g}" for v in (*p, *rpy(R)))


def geom_sdf(m, gid) -> tuple[str, str]:
    g = m.geom(gid)
    if g.type[0] == mujoco.mjtGeom.mjGEOM_BOX:
        shape = f"<box><size>{' '.join(f'{2 * v:.6g}' for v in g.size)}</size></box>"
    elif g.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        shape = f"<cylinder><radius>{g.size[0]:.6g}</radius><length>{2 * g.size[1]:.6g}</length></cylinder>"
    else:
        raise SystemExit(f"Tipo de geom no soportado: {g.name}")
    p = pose(m.geom_pos[gid], quat_mat(m.geom_quat[gid]))
    name = g.name
    color = METAL if "handle" in name or "latch" in name else WOOD
    mu = MU.get(name, 1.0)
    collision = (f"""      <collision name="{name}">
        <pose>{p}</pose>
        <geometry>{shape}</geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
          <contact><collide_bitmask>{BITMASK[name]}</collide_bitmask></contact>
        </surface>
      </collision>""")
    visual = (f"""      <visual name="{name}_visual">
        <pose>{p}</pose>
        <geometry>{shape}</geometry>
        <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
      </visual>""")
    return collision, visual


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(args.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mount = d.xpos[m.body("door_mount").id].copy()

    links = []
    for body, link in LINKS.items():
        bid = m.body(body).id
        R = d.xmat[bid].reshape(3, 3)
        link_pose = pose(RZ180 @ (d.xpos[bid] - mount), RZ180 @ R)
        Ri = quat_mat(m.body_iquat[bid])
        I = m.body_inertia[bid]
        geoms = [gid for gid in range(m.ngeom) if m.geom_bodyid[gid] == bid]
        parts = [geom_sdf(m, gid) for gid in geoms]
        links.append(f"""    <link name="{link}">
      <pose>{link_pose}</pose>
      <self_collide>true</self_collide>
      <inertial>
        <pose>{pose(m.body_ipos[bid], Ri)}</pose>
        <mass>{m.body_mass[bid]:.6g}</mass>
        <inertia><ixx>{I[0]:.6g}</ixx><iyy>{I[1]:.6g}</iyy><izz>{I[2]:.6g}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
{chr(10).join(c for c, _ in parts)}
{chr(10).join(v for _, v in parts)}
    </link>""")

    def joint(name, parent, child, extra):
        j = m.joint(name)
        lo, hi = m.jnt_range[j.id]
        return f"""    <joint name="{name}" type="revolute">
      <parent>{parent}</parent>
      <child>{child}</child>
      <pose>{' '.join(f'{v:.6g}' for v in m.jnt_pos[j.id])} 0 0 0</pose>
      <axis>
        <xyz>{' '.join(f'{v:.6g}' for v in m.jnt_axis[j.id])}</xyz>
        <limit><lower>{lo:.6g}</lower><upper>{hi:.6g}</upper></limit>
        <dynamics>{extra}</dynamics>
      </axis>
    </joint>"""

    hinge = m.joint("door_hinge").id
    handle = m.joint("handle_hinge").id
    sdf = f"""<?xml version="1.0" ?>
<!-- GENERADO por tools/door_sdf_from_mujoco.py desde {Path(args.xml).name}: no editar a mano.
     Origen = door_mount de MuJoCo; pose del fixture en Gazebo = ({-mount[0]:.4g}, {-mount[1]:.4g}, {mount[2]:.4g}). -->
<sdf version="1.7">
  <model name="door">
    <!-- Sin esto gz-sim no choca piezas del mismo modelo (el pestillo con su traba) -->
    <self_collide>true</self_collide>
    <!-- Marco fijo al mundo (sobre la mesa); la hoja y la manilla giran -->
    <joint name="anchor" type="fixed">
      <parent>world</parent>
      <child>frame</child>
    </joint>
{chr(10).join(links)}
{joint("door_hinge", "frame", "panel", f"<damping>{m.dof_damping[m.jnt_dofadr[hinge]]:.6g}</damping><friction>{m.dof_frictionloss[m.jnt_dofadr[hinge]]:.6g}</friction>")}
{joint("handle_hinge", "panel", "latch", f"<damping>{m.dof_damping[m.jnt_dofadr[handle]]:.6g}</damping><spring_reference>{m.qpos_spring[m.jnt_qposadr[handle]]:.6g}</spring_reference><spring_stiffness>{m.jnt_stiffness[handle]:.6g}</spring_stiffness>")}
    <!-- door_hinge / handle_hinge en /world/default/model/door/joint_state (éxito en eval) -->
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
  </model>
</sdf>
"""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.sdf").write_text(sdf, encoding="utf-8")
    (out / "model.config").write_text("""<?xml version="1.0"?>
<model>
  <name>door</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>Puerta con pestillo de la tarea Door Opening (generada desde MuJoCo)</description>
</model>
""", encoding="utf-8")
    print(f"{out / 'model.sdf'}\nPose del fixture (Gazebo): [{-mount[0]:.4g}, {-mount[1]:.4g}, {mount[2]:.4g}]")


if __name__ == "__main__":
    main()
