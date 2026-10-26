#!/usr/bin/env python3
"""Demo guionizada de la escena "Door Opening": el iCub abre la puerta con la mano derecha.

Sirve para comprobar que la puerta se puede abrir con el robot sin teleoperar (sin VR).
Usa el mismo núcleo que el teleop (MuJoCoTeleop: IK iKinIpOptMin, interpolación min-jerk
y apertura/cierre de mano del YAML); solo cambia quién mueve el target de la mano:

  1. pre-agarre: mano sobre la manilla, palma hacia abajo y dedos hacia la puerta
  2. baja hasta la manilla y cierra la mano (engancha la barra)
  3. baja la manilla para soltar el pestillo
  4. tira siguiendo el arco de la puerta alrededor de la bisagra
  5. abre la mano, se retira e informa el ángulo final de door_hinge

Uso (entorno con iKin, p. ej. icubenv):
    python demo_door_mujoco.py              # con visor
    python demo_door_mujoco.py --headless   # sin visor, solo imprime el resultado
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

import dependencies.teleop_mujoco as tm  # noqa: E402
from dependencies.door_scene import place_door, set_handle_side  # noqa: E402

SCENE = repo_root / "dependencies" / "assets" / "scenes" / "scene_icub_door.xml"
CONFIG = repo_root / "utils" / "control_config.yaml"


def rot_axis(axis, angle):
    """Matriz de rotación de `angle` rad alrededor de `axis` (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


def build_plan(m, d, args, arm="right"):
    """Waypoints [(duración s, pos, R, mano 0..1, etiqueta)] desde la geometría de la escena.

    Todo en el mundo MuJoCo y con la escena en su estado actual (puerta cerrada, robot en
    home): la mano parte de donde está y cada tramo interpola hasta su waypoint.
    """
    g = m.geom("handle_collision").id
    bar = d.geom_xpos[g].copy()
    bar_axis = d.geom_xmat[g].reshape(3, 3)[:, 0]                 # largo de la barra
    toward_robot = bar - d.xpos[m.body("latch").id]                # la manilla sobresale hacia el robot
    toward_robot -= np.dot(toward_robot, bar_axis) * bar_axis
    toward_robot[2] = 0.0
    toward_robot /= np.linalg.norm(toward_robot)
    # La palanca sale del eje de la manilla hacia la bisagra; grip_shift corre el agarre por la
    # barra en ese sentido (hacia la punta, más cerca del brazo contrario).
    if np.dot(bar - d.xpos[m.body("latch").id], bar_axis) < 0:
        bar_axis = -bar_axis
    bar = bar + getattr(args, "grip_shift", 0.0) * bar_axis

    # Marco del sitio <r|l>_hand_dh_frame: dedos en +x local y pulgar en -y local; la mano
    # cierra hacia la palma, que es +z local en la derecha y -z local en la izquierda.
    up = np.array([0.0, 0.0, 1.0])
    if args.grasp == "top":
        # Desde arriba: dedos hacia la puerta, palma hacia abajo, sobre la barra
        x_l, palm = -toward_robot, -up
    else:
        # De frente: dedos hacia arriba, palma hacia la puerta, bajo la barra (los dedos
        # pasan por encima y se enganchan por detrás al cerrar)
        x_l, palm = up, -toward_robot
    z_l = palm if arm == "right" else -palm
    R_grasp = np.column_stack([x_l, np.cross(z_l, x_l), z_l])

    grasp = bar + args.back * toward_robot + args.up * up
    pre = grasp + 0.06 * toward_robot + (0.06 * up if args.grasp == "top" else 0.0)

    # Pestillo: bajar la manilla girando la mano alrededor de su eje (signo tal que la barra
    # baje) hasta press_deg; así latch_tip pasa sobre latch_strike.
    jh = m.joint("handle_hinge").id
    h_anchor, h_axis = d.xanchor[jh].copy(), d.xaxis[jh].copy()
    if (rot_axis(h_axis, -0.1) @ (bar - h_anchor))[2] > (bar - h_anchor)[2]:
        h_axis = -h_axis
    Rp = rot_axis(h_axis, -np.radians(args.press_deg))
    pressed, R_pressed = h_anchor + Rp @ (grasp - h_anchor), Rp @ R_grasp

    j = m.joint("door_hinge").id
    hinge, axis = d.xanchor[j].copy(), d.xaxis[j].copy()
    pull = []
    n = 12
    for k in range(1, n + 1):
        th = np.radians(args.open_deg) * k / n
        Rh = rot_axis(axis, th)
        pull.append((args.pull_s / n, hinge + Rh @ (pressed - hinge),
                     rot_axis(axis, args.twist * th) @ R_pressed, 1.0, "tira"))
    end_pos, end_R = pull[-1][1], pull[-1][2]
    retreat = end_pos + 0.08 * (rot_axis(axis, np.radians(args.open_deg)) @ toward_robot)
    # En home la mano queda justo delante de la manilla: primero sube, para no pasar a
    # llevar la palanca (y soltar el pestillo) camino al pre-agarre.
    site = m.site(f"{arm[0]}_hand_dh_frame_site").id
    lift = d.site_xpos[site] + np.array([0.0, 0.0, 0.08]) + 0.03 * toward_robot
    R_home = d.site_xmat[site].reshape(3, 3).copy()
    return ([(1.5, lift, R_home, 0.0, "sube la mano"),
             (2.5, pre, R_grasp, 0.0, "pre-agarre"),
             (1.5, grasp, R_grasp, 0.0, "baja a la manilla"),
             (args.close_s, grasp, R_grasp, 1.0, "cierra la mano"),
             (args.press_s, pressed, R_pressed, 1.0, "baja la manilla")]
            + pull
            + [(0.5, end_pos, end_R, 1.0, "sostiene"),
               (args.close_s, end_pos, end_R, 0.0, "abre la mano"),
               (1.5, retreat, end_R, 0.0, "se retira")]), bar


def idle_plan(m, d, arm, back=0.10, out=0.06, down=0.03, dur=1.5):
    """Mano que no abre la puerta: se aparta del arco de la hoja (hacia atrás, hacia su lado y
    un poco abajo) y se queda ahí abierta. En home la hoja, al abrirse, le pega."""
    site = m.site(f"{arm[0]}_hand_dh_frame_site").id
    outward = np.array([0.0, 1.0 if arm == "left" else -1.0, 0.0])
    pos = d.site_xpos[site] + np.array([-back, 0.0, -down]) + out * outward
    return [(dur, pos, d.site_xmat[site].reshape(3, 3).copy(), 0.0, "aparta la otra mano")]


class Trajectory:
    """Target de la mano en función del tiempo: interpola (lineal + slerp) entre waypoints."""

    def __init__(self, plan, p_start, q_start):
        self.plan = plan
        self.starts = [(np.asarray(p_start, float), np.asarray(q_start, float))]
        for dur, pos, R, hand, label in plan[:-1]:
            self.starts.append((pos, mat2quat(R)))
        self.t_end = np.cumsum([wp[0] for wp in plan])
        self.duration = float(self.t_end[-1])

    def shift(self, first, delta):
        """Corre los waypoints desde el índice `first` en `delta` (corrección en lazo cerrado)."""
        for i in range(first, len(self.plan)):
            dur, pos, R, hand, label = self.plan[i]
            self.plan[i] = (dur, pos + delta, R, hand, label)
            if i + 1 < len(self.starts):
                self.starts[i + 1] = (pos + delta, self.starts[i + 1][1])

    def at(self, t):
        """(pos, quat wxyz, mano, etiqueta, índice del tramo) en el tiempo t [s]."""
        i = min(int(np.searchsorted(self.t_end, t, side="right")), len(self.plan) - 1)
        dur, pos, R, hand, label = self.plan[i]
        s = float(np.clip(1.0 - (self.t_end[i] - t) / dur, 0.0, 1.0))
        p0, q0 = self.starts[i]
        return p0 + s * (pos - p0), slerp(q0, mat2quat(R), s), hand, label, i


def add_plan_args(p):
    p.add_argument("--open-deg", type=float, default=55.0, help="Ángulo de apertura a seguir con la mano")
    p.add_argument("--pull-s", type=float, default=7.0, help="Duración del tirón [s]")
    p.add_argument("--press-deg", type=float, default=40.0,
                   help="Cuánto baja la mano la manilla [deg] (el pestillo suelta desde ~19 deg)")
    p.add_argument("--press-s", type=float, default=1.5, help="Duración de bajar la manilla [s]")
    p.add_argument("--close-s", type=float, default=3.5, help="Tiempo para cerrar/abrir la mano [s]")
    p.add_argument("--grasp", choices=["top", "front"], default="top",
                   help="top: palma abajo, dedos hacia la puerta | front: palma hacia la puerta, dedos arriba")
    p.add_argument("--back", type=float, default=None,
                   help="Palma respecto de la barra, hacia el robot [m] (top 0.04, front 0.02)")
    p.add_argument("--up", type=float, default=None,
                   help="Palma respecto de la barra, hacia arriba [m] (top 0.01, front -0.04)")
    p.add_argument("--twist", type=float, default=1.0,
                   help="Fracción del giro de la puerta que acompaña la muñeca (1 = gira con la manilla)")
    p.add_argument("--grip-shift", type=float, default=0.0,
                   help="Agarre corrido por la barra hacia su punta (lado de la bisagra) [m]")


def fill_plan_defaults(args):
    if args.back is None:
        args.back = 0.02 if args.grasp == "front" else 0.04
    if args.up is None:
        args.up = -0.04 if args.grasp == "front" else 0.01
    return args


def door_open(m, d):
    """Éxito como en robosuite (Door._check_success): bisagra > 0.3 rad (~17 deg)."""
    return float(d.qpos[m.jnt_qposadr[m.joint("door_hinge").id]]) > 0.3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(SCENE))
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--headless", action="store_true", help="Sin visor (más rápido que tiempo real)")
    add_plan_args(p)
    p.add_argument("--grip", type=float, default=None,
                   help="Fracción de cierre de la mano (por defecto hand_close_ratio del YAML)")
    p.add_argument("--arm", choices=["right", "left"], default="right", help="Mano que abre la puerta")
    p.add_argument("--control-arms", choices=["right", "left", "both"], default=None,
                   help="Brazos con IK (por defecto: right con --arm right, both con --arm left). "
                        "El torso es del brazo derecho (primary_arm), como en el Hub")
    p.add_argument("--side", choices=["right", "left"], default=None,
                   help="Lado de la manilla (por defecto el del XML: izquierda)")
    p.add_argument("--door-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="Centro de la hoja en el mundo (mesa: 0.5 0); por defecto el del XML")
    args = fill_plan_defaults(p.parse_args())
    control_arms = args.control_arms or ("right" if args.arm == "right" else "both")

    tm.BEAVR_IMG_AVAILABLE = False  # sin streaming a los lentes (no abre puertos ZMQ)
    t = tm.MuJoCoTeleop(args.model, config_path=args.config, control_arms=control_arms, primary_arm="right")
    m, d = t.model, t.data
    if args.side is not None:
        set_handle_side(m, args.side)
    if args.door_xy is not None:
        mujoco.mj_forward(m, d)
        place_door(m, d, args.door_xy)
    if args.grip is not None:
        t._hand_close_goal = t._hand_open + float(np.clip(args.grip, 0, 1)) * (t._hand_close_full - t._hand_open)
    d.qpos[:] = t.home_qpos
    d.ctrl[:] = t.home_ctrl
    t._apply_home_pose_to_arrays(d.qpos, d.ctrl)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(int(1.0 / t.control_dt) * t.frame_skip):   # 1 s para asentarse en home
        mujoco.mj_step(m, d)
    t._snap_targets_to_ee()

    plan, bar = build_plan(m, d, args, arm=args.arm)
    mocap = t.rh_mocap_id if args.arm == "right" else t.lh_mocap_id
    site = t.rh_site_id if args.arm == "right" else t.lh_site_id
    traj = Trajectory(plan, d.mocap_pos[mocap], d.mocap_quat[mocap])
    idle_arm = "left" if args.arm == "right" else "right"
    idle_mocap = t.lh_mocap_id if idle_arm == "left" else t.rh_mocap_id
    idle_traj = (Trajectory(idle_plan(m, d, idle_arm), d.mocap_pos[idle_mocap], d.mocap_quat[idle_mocap])
                 if control_arms == "both" else None)
    d.mocap_pos[t.gaze_mocap_id] = bar
    hinge_adr = m.jnt_qposadr[m.joint("door_hinge").id]
    handle_adr = m.jnt_qposadr[m.joint("handle_hinge").id]

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
        viewer.cam.lookat[:] = [0.3, 0.0, 0.8]
        viewer.cam.distance = 1.3
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 200

    def report(i):
        err = np.linalg.norm(d.site_xpos[site] - plan[i][1]) * 100
        print(f"        mano a {err:4.1f} cm del target   door_hinge = {np.degrees(d.qpos[hinge_adr]):5.1f} deg"
              f"   handle_hinge = {np.degrees(d.qpos[handle_adr]):5.1f} deg")

    t0_sim, seg, max_door = d.time, -1, 0.0
    while d.time - t0_sim < traj.duration:
        pos, quat, hand, label, i = traj.at(d.time - t0_sim)
        if i != seg:
            if seg >= 0:
                report(seg)
            if seg < 0 or label != plan[seg][4]:
                print(f"[demo] {label}")
            seg = i
            t._set_hand_level(args.arm, hand)
        d.mocap_pos[mocap] = pos
        d.mocap_quat[mocap] = quat
        if idle_traj is not None:
            d.mocap_pos[idle_mocap], d.mocap_quat[idle_mocap] = idle_traj.at(d.time - t0_sim)[:2]
        t_wall = time.time()
        t._solve_ik()
        t._interpolate_hands()
        for _ in range(t.frame_skip):
            mujoco.mj_step(m, d)
        max_door = max(max_door, float(d.qpos[hinge_adr]))
        if viewer is not None:
            if not viewer.is_running():
                return
            viewer.sync()
            time.sleep(max(0.0, t.control_dt - (time.time() - t_wall)))
    report(seg)

    final = float(d.qpos[hinge_adr])
    print(f"\n[demo] door_hinge final = {np.degrees(final):.1f} deg (máximo {np.degrees(max_door):.1f} deg) "
          f"-> {'ABIERTA' if door_open(m, d) else 'NO se abrió'} (éxito robosuite: > 0.3 rad = 17 deg)")
    if viewer is not None:
        print("[demo] Cierra la ventana del visor para salir.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()


if __name__ == "__main__":
    main()
