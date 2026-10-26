#!/usr/bin/env python3
"""Copia un dataset de Gazebo pasando la acción de pinza de pulsos a nivel.

Los datasets de Gazebo con la pinza como pulsos tienen 1.0 ("nada") casi siempre y un solo
frame de 0.5 (cerrar) o 0.0 (abrir) al apretar el botón. Una
política entrenada con eso aprende a sacar 1.0 y nunca abre ni cierra la mano.

Por defecto la copia usa la convención actual (gatillo gradual):
rh/lh_gripper = fracción de cierre, 0.0 abierta (también antes del primer comando) y 1.0
cerrada. Con --legacy-levels usa la intermedia: 1.0 hasta el primer comando, luego 0.0
abierta / 0.5 cerrada (con la que se entrenó gazebotrainPMM). Recalcula las estadísticas
de la acción (meta/stats.json y las de cada episodio). El original no se toca.

Uso:
    python tools/gripper_pulses_to_level.py ../data/local_icub_gazebo_2026-10-15_15-08-04 \\
        [--dst ../data_curated/<nombre>] [--task "lift the blue block"]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

GRIPPER_KEYS = ("rh_gripper", "lh_gripper")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def pulses_to_level(values: np.ndarray, legacy: bool = False) -> np.ndarray:
    """Umbrales de los pulsos: <=0.1 abrir, 0.4-0.6 cerrar, resto = mantener el nivel."""
    closed = 0.5 if legacy else 1.0
    level, out = (1.0 if legacy else 0.0), np.empty_like(values)
    for i, v in enumerate(values):
        if v <= 0.1:
            level = 0.0
        elif 0.4 <= v <= 0.6:
            level = closed
        out[i] = level
    return out


def stats_of(actions: np.ndarray, keys) -> dict:
    stats = {
        "min": actions.min(0), "max": actions.max(0), "mean": actions.mean(0),
        "std": actions.std(0), "count": np.array([len(actions)]),
    }
    stats.update({k: np.quantile(actions, q, axis=0) for k, q in QUANTILES.items()})
    return {k: stats[k] for k in keys}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="Dataset root (with meta/info.json)")
    p.add_argument("--dst", type=Path, default=None, help="Output root (default: <src>_gripper_level)")
    p.add_argument("--task", default=None, help="Replace the (single) task text, e.g. to match MuJoCo")
    p.add_argument("--legacy-levels", action="store_true",
                   help="Old levels 1 (none yet) / 0 open / 0.5 closed instead of the closure fraction")
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    dst = (args.dst or src.with_name(src.name + "_gripper_level")).expanduser().resolve()
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("meta_backup*"))

    info = json.loads((dst / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    cols = [names.index(k) for k in GRIPPER_KEYS if k in names]
    if not cols:
        raise SystemExit(f"No gripper in action names: {names}")

    # Datos: convertir por episodio, en orden de frame
    all_actions, per_episode = [], {}
    for f in sorted((dst / "data").glob("**/*.parquet")):
        df = pd.read_parquet(f)
        actions = np.stack(df["action"].values).astype(np.float32)
        for ep in df["episode_index"].unique():
            rows = np.flatnonzero(df["episode_index"].values == ep)
            rows = rows[np.argsort(df["frame_index"].values[rows])]
            for c in cols:
                actions[rows, c] = pulses_to_level(actions[rows, c], args.legacy_levels)
            per_episode[int(ep)] = actions[rows]
        df["action"] = list(actions)
        df.to_parquet(f, index=False)
        all_actions.append(actions)
    all_actions = np.concatenate(all_actions)

    # Estadísticas globales (normalización del entrenamiento)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = {k: v.tolist() for k, v in stats_of(all_actions, stats["action"].keys()).items()}
    stats_path.write_text(json.dumps(stats, indent=4))

    # Estadísticas por episodio (meta/episodes)
    for f in sorted((dst / "meta" / "episodes").glob("**/*.parquet")):
        ep_df = pd.read_parquet(f)
        keys = [c.removeprefix("stats/action/") for c in ep_df.columns if c.startswith("stats/action/")]
        new_cols = {k: [] for k in keys}
        for row, ep in enumerate(ep_df["episode_index"].values):
            ep_stats = stats_of(per_episode[int(ep)], keys)
            for k in keys:
                old = np.asarray(ep_df[f"stats/action/{k}"].iloc[row])  # mismo tipo y forma
                new_cols[k].append(np.asarray(ep_stats[k], dtype=old.dtype).reshape(old.shape))
        for k, values in new_cols.items():
            ep_df[f"stats/action/{k}"] = values
        ep_df.to_parquet(f, index=False)

    if args.task:
        tasks = pd.read_parquet(dst / "meta" / "tasks.parquet")
        if len(tasks) != 1:
            raise SystemExit(f"--task needs a single-task dataset, found {len(tasks)}")
        tasks.index = [args.task]
        tasks.index.name = "task"
        tasks.to_parquet(dst / "meta" / "tasks.parquet")

    for c in cols:
        values, counts = np.unique(all_actions[:, c], return_counts=True)
        print(f"{names[c]}: " + ", ".join(f"{v:g} -> {n} frames" for v, n in zip(values, counts)))
    print(f"Saved to {dst}")


if __name__ == "__main__":
    main()
