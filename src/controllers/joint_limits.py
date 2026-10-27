"""Joint limits of the 20 upper-body joints: `joint_limits_deg` in config/control_config.yaml.

It is the only source of the limits. They are applied to:
  - the whole-body solver (iCubWholeBodyIK: JointLimitsTask; iCubWholeBodyID: joint
    acceleration bounds that brake before the limit),
  - MuJoCo: joint ranges and position actuators' ctrlrange, overwritten when the model loads
    (MuJoCoTeleop), so the XML ranges do not matter,
  - Gazebo: the controlboards' limits (BLFRobotBridge.apply_joint_limits, from the YARP module).
On the real robot only the solver uses them: its firmware limits are not touched.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Joints whose limits are configurable (the ones the whole-body solver controls)
UPPER_BODY_JOINTS = (
    "torso_pitch", "torso_roll", "torso_yaw",
    "neck_pitch", "neck_roll", "neck_yaw",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
    "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
)


def load_joint_limits(cfg: dict) -> Dict[str, Tuple[float, float]]:
    """{joint: (min_rad, max_rad)} from cfg["joint_limits_deg"]; {} (model limits) if absent."""
    raw = cfg.get("joint_limits_deg") or {}
    if not raw:
        print("[joint_limits] joint_limits_deg not found in control_config.yaml: model limits are used.")
        return {}
    unknown = sorted(set(raw) - set(UPPER_BODY_JOINTS))
    if unknown:
        raise ValueError(f"joint_limits_deg: unknown joints {unknown} (expected {list(UPPER_BODY_JOINTS)})")
    limits = {}
    for name, pair in raw.items():
        lo, hi = (float(v) for v in pair)
        if lo >= hi:
            raise ValueError(f"joint_limits_deg.{name}: min {lo} must be < max {hi}")
        limits[name] = (float(np.deg2rad(lo)), float(np.deg2rad(hi)))
    return limits
