#!/usr/bin/env python3
"""Standardized Control & Solver Metadata Management for iCub Datasets and Policies.

Manages serialization, strict validation, and propagation of whole-body controller
metadata (`meta/control_config.json` in datasets and `control_config.json` in checkpoints).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import yaml


SCHEMA_VERSION = "1.0"


def generate_standard_dataset_name(
    robot: str = "icub",
    sim: str = "mujoco",
    controller: str = "tsik",
    mode: str = "position-direct",
    task_label: str = "block-stacking",
    num_episodes: int = 50,
    org: str | None = None,
) -> str:
    """Generate a deterministic, standardized dataset repository name without 'local/' prefix.
    
    Format:
      [<org>/]<robot>_<sim>_<controller>_<mode>_<task>_<demos>eps
    """
    r = str(robot).strip().lower()
    s = str(sim).strip().lower()
    ctrl = "tsik" if str(controller).strip().lower() in ("ik", "tsik") else "tsid"
    m = str(mode).strip().lower().replace("_", "-")
    slug = str(task_label).strip().lower().replace(" ", "-").replace("_", "-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    eps = int(num_episodes)
    base = f"{r}_{s}_{ctrl}_{m}_{slug}_{eps}eps"
    if org and str(org).strip():
        return f"{str(org).strip()}/{base}"
    return base


def normalize_controller_type(controller: str) -> str:
    """Normalize controller name to 'tsik' or 'tsid'."""
    low = str(controller).strip().lower()
    if low in ("ik", "tsik"):
        return "tsik"
    if low in ("id", "tsid"):
        return "tsid"
    raise ValueError(f"Invalid controller type '{controller}'. Expected 'tsik' ('ik') or 'tsid' ('id').")


def resolve_control_mode(
    controller_type: str,
    impedance: bool = False,
    yarp_mode: str | None = None,
) -> str:
    """Resolve exact control mode.
    
    For TSID:
      - 'torque' (direct torque control, nominal)
      - 'impedance' (compliant Cartesian impedance)
    For TSIK:
      - 'position_direct' (nominal direct position streaming from integrated QP velocities)
      - 'position' (minimum-jerk / profiler trajectory generator)
      - 'velocity' (closed-loop joint velocity control)
      - 'velocity_direct' (direct velocity streaming)
    """
    ctrl = normalize_controller_type(controller_type)
    if ctrl == "tsid":
        return "impedance" if impedance else "torque"

    # TSIK modes
    ym = str(yarp_mode).strip().lower() if yarp_mode else "position_direct"
    valid_tsik_modes = ("position_direct", "position", "velocity", "velocity_direct")
    if ym not in valid_tsik_modes:
        ym = "position_direct"
    return ym


def load_raw_control_config_yaml(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load raw control_config.yaml."""
    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "control_config.yaml"
    p = Path(config_path).expanduser().resolve()
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_control_metadata(
    controller: str,
    impedance: bool = False,
    mode: str | None = None,
    config_yaml_path: str | Path | None = None,
    robot_type: str = "icub",
    robot_model: str | None = None,
    robot_name: str | None = None,
    robot_yarp_name: str | None = None,
    arms: str = "both",
    use_gaze: bool = True,
    backend: str = "mujoco",
    scene: str | None = None,
    task: str | None = None,
    scene_file: str | None = None,
    fps: int = 30,
    sim_timestep: float | None = None,
    sim_fps: int | None = None,
    total_episodes: int | None = None,
    repo_id: str | None = None,
    user_notes: str | None = None,
) -> dict[str, Any]:
    """Construct a complete, standardized control metadata dictionary."""
    ctrl_type = normalize_controller_type(controller)
    raw_cfg = load_raw_control_config_yaml(config_yaml_path)

    effective_mode = (
        mode
        if mode
        else resolve_control_mode(
            controller_type=ctrl_type,
            impedance=impedance,
            yarp_mode=raw_cfg.get("yarp_control_mode"),
        )
    )

    # Extract solver parameters corresponding to the active controller
    solver_params: dict[str, Any] = {}
    if ctrl_type == "tsid":
        solver_params["tsid"] = dict(raw_cfg.get("tsid", {}))
    else:
        solver_params["tsik"] = dict(raw_cfg.get("tsik", {}))

    joint_limits = raw_cfg.get("joint_limits_deg")
    if joint_limits:
        solver_params["joint_limits_deg"] = dict(joint_limits)

    b = str(backend).strip().lower()
    if b == "mujoco":
        effective_model = robot_model or robot_name or "iCub"
        effective_yarp = None
        effective_sim_timestep = 0.005 if sim_timestep is None else float(sim_timestep)
        effective_sim_fps = 200 if sim_fps is None else int(sim_fps)
    elif b == "gazebo":
        effective_model = robot_model or raw_cfg.get("robot_model", "iCubGazeboV2_5_visuomanip")
        effective_yarp = robot_yarp_name or os.environ.get("ICUB_ROBOT_NAME", "icubSim")
        effective_sim_timestep = 0.005 if sim_timestep is None else float(sim_timestep)
        effective_sim_fps = 200 if sim_fps is None else int(sim_fps)
    elif b == "real":
        effective_model = robot_model or "iCub2.7"
        effective_yarp = robot_yarp_name or os.environ.get("ICUB_ROBOT_NAME", "icub")
        effective_sim_timestep = None
        effective_sim_fps = None
    else:
        effective_model = robot_model or robot_name or "iCub"
        effective_yarp = robot_yarp_name
        effective_sim_timestep = sim_timestep
        effective_sim_fps = sim_fps

    ctrl_fps = int(fps) if fps else 30
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot": {
            "type": "icub",
            "model": effective_model,
            "name": effective_model,  # backwards-compatible alias
            "yarp_name": effective_yarp,
            "arms": arms,
            "use_gaze": bool(use_gaze),
        },
        "simulation": {
            "backend": b,
            "scene": scene or "Unknown",
            "task": task or "",
            "scene_file": scene_file or "",
            "control_fps": ctrl_fps,
            "control_dt": 1.0 / float(ctrl_fps),
            "sim_timestep": effective_sim_timestep,
            "sim_fps": effective_sim_fps,
            "fps": ctrl_fps,  # backwards-compatible alias
        },
        "control": {
            "type": ctrl_type,
            "mode": effective_mode,
            "impedance_enabled": bool(impedance),
            "source_config_file": str(config_yaml_path) if config_yaml_path else "config/control_config.yaml",
        },
        "parameters": solver_params,
    }

    if repo_id:
        metadata["repo_id"] = str(repo_id).strip()

    recording_info: dict[str, Any] = {}
    if total_episodes is not None:
        recording_info["total_episodes"] = int(total_episodes)
    if user_notes:
        recording_info["user_notes"] = str(user_notes)
    if recording_info:
        metadata["recording"] = recording_info

    return metadata


def save_dataset_control_metadata(dataset_root: Path | str, metadata: dict[str, Any]) -> Path:
    """Save metadata to `meta/control_config.json` inside dataset directory."""
    root = Path(dataset_root).expanduser().resolve()
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_file = meta_dir / "control_config.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[control_metadata] Saved dataset control metadata: {out_file}")
    return out_file


def load_dataset_control_metadata(dataset_root: Path | str) -> dict[str, Any]:
    """Strictly load and validate `meta/control_config.json` from a dataset."""
    root = Path(dataset_root).expanduser().resolve()
    meta_file = root / "meta" / "control_config.json"
    if not meta_file.is_file():
        raise FileNotFoundError(
            f"Dataset at '{root}' is missing required 'meta/control_config.json'. "
            "Legacy uncharacterized datasets without control metadata are strictly not allowed."
        )
    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "control" not in data or "type" not in data["control"]:
        raise ValueError(f"Corrupt or invalid control metadata in {meta_file}: missing 'control.type'")
    return data


def load_checkpoint_control_metadata(checkpoint_dir: Path | str) -> dict[str, Any]:
    """Strictly load and validate `control_config.json` from a model checkpoint."""
    root = Path(checkpoint_dir).expanduser().resolve()
    cfg_file = root / "control_config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint at '{root}' is missing required 'control_config.json'. "
            "Legacy uncharacterized models without control metadata are strictly not allowed."
        )
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "control" not in data or "type" not in data["control"]:
        raise ValueError(f"Corrupt or invalid control metadata in {cfg_file}: missing 'control.type'")
    return data


def find_model_control_config(checkpoint_dir: Path | str, dataset_meta: Path | str | None = None) -> Path | None:
    """Locate the control metadata of a trained model, without copying it at training time.

    Search order: `control_config.json` in the checkpoint, then `meta/control_config.json` of the
    dataset it was trained on (`dataset.root` in the checkpoint's `train_config.json`), then
    `meta/control_config.json` of `dataset_meta`. Returns None if none exists.
    """
    ckpt = Path(checkpoint_dir).expanduser()
    candidates = [ckpt / "control_config.json"]
    try:
        root = json.loads((ckpt / "train_config.json").read_text(encoding="utf-8"))["dataset"]["root"]
        if root:
            candidates.append(Path(root).expanduser() / "meta" / "control_config.json")
    except (OSError, KeyError, TypeError, ValueError):
        pass
    if dataset_meta:
        candidates.append(Path(dataset_meta).expanduser().resolve() / "meta" / "control_config.json")
    return next((c for c in candidates if c.is_file()), None)


def controller_from_metadata(metadata: dict[str, Any]) -> tuple[str, bool]:
    """(controller, impedance) for the runtime: 'id' for TSID, 'ik' otherwise; impedance only with ID."""
    ctrl = metadata.get("control", {})
    controller = "id" if str(ctrl.get("type", "tsid")).lower() in ("id", "tsid") else "ik"
    return controller, controller == "id" and bool(ctrl.get("impedance_enabled", False))


def copy_control_metadata_to_checkpoint(
    source_dataset_root_or_file: Path | str,
    target_checkpoint_dir: Path | str,
) -> Path:
    """Copy `meta/control_config.json` into a target checkpoint directory as `control_config.json`."""
    src = Path(source_dataset_root_or_file).expanduser().resolve()
    src_file = src if src.is_file() else src / "meta" / "control_config.json"
    if not src_file.is_file():
        raise FileNotFoundError(f"Cannot copy control metadata: source '{src_file}' does not exist.")

    dst_dir = Path(target_checkpoint_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / "control_config.json"
    shutil.copy2(src_file, dst_file)
    print(f"[control_metadata] Propagated control metadata to checkpoint: {dst_file}")
    return dst_file

