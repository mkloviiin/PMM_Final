"""Selector del controlador whole-body (BLF), compartido por los dos backends del iCub.

Las tres opciones son las de config/control_config.yaml (controller_type + impedance_mode):
  - IK  : iCubWholeBodyIK (TSIK), comanda posiciones (MuJoCo: actuadores de posición; Gazebo /
          real: POSITION_DIRECT).
  - ID  : iCubWholeBodyID (TSID), comanda torques (MuJoCo: icub_visuomanip_idyn_limits.xml;
          Gazebo / real: modo TORQUE), seguimiento rígido de los targets.
  - ID + impedancia: el mismo TSID con perfil cartesiano compliante.
En MuJoCo lo recibe play.py; en Gazebo / real, el módulo YARP (paso 2.2), que es quien corre el
solver, y también play.py (activa o no la impedancia al empezar el teleop).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import yaml

SOLVER_IK = "Whole-Body IK (position)"
SOLVER_ID = "Whole-Body ID (torque)"
SOLVER_ID_IMP = "Whole-Body ID (torque) + impedance"
# label -> (controller_type, impedance_mode)
SOLVERS = {
    SOLVER_IK: ("ik", False),
    SOLVER_ID: ("id", False),
    SOLVER_ID_IMP: ("id", True),
}


def _yaml_solver(cfg_path: Path) -> str:
    """Opción que corresponde al controller_type / impedance_mode del YAML (valor inicial)."""
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return SOLVER_IK
    key = (str(cfg.get("controller_type", "ik")).lower(), bool(cfg.get("impedance_mode", False)))
    return next((label for label, v in SOLVERS.items() if v == key), SOLVER_IK)


def solver_dropdown(cfg_path: Path, info: str) -> gr.Dropdown:
    return gr.Dropdown(
        choices=list(SOLVERS), value=_yaml_solver(cfg_path),
        label="Controller (solver)", info=info,
    )


def solver_argv(label: str) -> list[str]:
    """Argumentos de play.py / teleop_module_sm.py para la opción elegida."""
    controller, impedance = SOLVERS.get(label, SOLVERS[SOLVER_IK])
    return ["--controller", controller] + (["--impedance"] if impedance else [])
