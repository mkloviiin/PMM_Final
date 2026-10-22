#!/usr/bin/env python3
"""Mide dónde queda el cubo en el frame de la mano al agarrar, en dos datasets, y su diferencia.

En cada episodio que levanta el cubo se toma el último cierre de la pinza (rh_gripper > 0.5,
el que funcionó) y se expresa la posición del cubo en el frame de la mano (rh_ee_pos /
rh_ee_quat, frame DH de iKin en ambos simuladores). La diferencia eval - train es el
--grasp-offset de eval_gazebo.py: con la misma pose del frame de la mano, los dedos de cada
simulador necesitan el cubo en otro lugar. Si las manos tienen la misma geometría, ≈ 0.

Uso:
    python tools/grasp_offset.py \\
        --train ../data/local_final_mujoco_2026-10-20_14-53-16 \\
        --eval  ../data/local_final_gazebo_2026-10-20_15-25-43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POS = [f"rh_ee_pos_{ax}" for ax in "xyz"]
QUAT = [f"rh_ee_quat_{ax}" for ax in "wxyz"]
OBJ = [f"obj0_pos_{ax}" for ax in "xyz"]


def quat_rotate_inv(quat, vec) -> np.ndarray:
    """Rota vec por el inverso del cuaternión (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    u = -np.array([x, y, z])
    v = np.asarray(vec, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def cube_in_hand(root: Path, lift_m: float) -> np.ndarray:
    """Cubo en el frame de la mano al iniciar el último cierre de cada episodio que levanta."""
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    missing = [k for k in POS + QUAT + OBJ if k not in names]
    if missing:
        raise SystemExit(f"{root}: state lacks {missing} (needs use_eef_pose and object poses)")
    grip_idx = info["features"]["action"]["names"].index("rh_gripper")
    files = sorted((root / "data").glob("**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    state = pd.DataFrame(np.stack(df["observation.state"].values), columns=names)
    grip = np.stack(df["action"].values)[:, grip_idx]

    out = []
    for _, idx in df.groupby("episode_index").indices.items():
        s, closed = state.iloc[idx], grip[idx] > 0.5
        if s[OBJ[2]].max() - s[OBJ[2]].iloc[0] < lift_m or not closed.any():
            continue
        k = np.flatnonzero(np.diff(np.r_[0, closed.astype(int)]) == 1)[-1]
        row = s.iloc[k]
        out.append(quat_rotate_inv(row[QUAT].values, row[OBJ].values - row[POS].values))
    if not out:
        raise SystemExit(f"{root}: no episode lifts the cube by {lift_m} m")
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", type=Path, required=True, help="Dataset of the simulator the policy was trained on")
    p.add_argument("--eval", type=Path, required=True, help="Dataset of the simulator it is evaluated on")
    p.add_argument("--lift-m", type=float, default=0.03, help="Minimum cube lift to count an episode [m]")
    args = p.parse_args()

    means = {}
    for label, root in (("train", args.train), ("eval", args.eval)):
        c = cube_in_hand(root.expanduser().resolve(), args.lift_m)
        means[label] = c.mean(0)
        print(f"{label:5s} {root.name}: n={len(c)}  cube in hand frame = "
              f"{np.round(c.mean(0) * 100, 1).tolist()} cm  (std {np.round(c.std(0) * 100, 1).tolist()})")
    offset = means["eval"] - means["train"]
    print(f"\n--grasp-offset {offset[0]:.3f} {offset[1]:.3f} {offset[2]:.3f}   "
          f"(|offset| = {np.linalg.norm(offset) * 100:.1f} cm)")


if __name__ == "__main__":
    main()
