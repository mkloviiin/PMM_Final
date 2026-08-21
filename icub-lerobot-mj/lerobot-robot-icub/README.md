# lerobot_robot_icub

LeRobot robot plugin for iCub through YARP.

## Registered robot types

- `lerobot_robot_icub`

## What it implements

- `RobotConfig` subclass: `iCubConfig`
- `Robot` subclass: `iCub`
- observation/action feature contract
- `connect`, `disconnect`, `get_observation`, `send_action`

## Install

```bash
pip install .
```

## Runtime notes

- Requires YARP runtime and Python bindings (`import yarp`) in the same environment.
- Defaults are compatible with `icub-lerobot/smteleop/teleop_module_sm.py` ports.
- Set config path with:

```bash
export ICUB_LEROBOT_CONFIG=/home/icub/mujoco_ws/icub-lerobot/utils/control_config.yaml
```
