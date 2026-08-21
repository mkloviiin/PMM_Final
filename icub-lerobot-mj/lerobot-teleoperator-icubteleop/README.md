# lerobot_teleoperator_icubteleop

LeRobot teleoperator plugin for iCub using MuJoCo mocap interaction.

## Registered teleoperator types

- `icub_teleop_mujoco`
- `lerobot_teleoperator_icubteleop`

## What it implements

- `TeleoperatorConfig` subclass: `iCubTeleopConfig`
- `Teleoperator` subclass: `iCubTeleop`
- `get_action` for teleoperation commands
- `send_feedback` to visualize robot feedback in MuJoCo

## Install

```bash
pip install -e .
```

## Runtime notes

- Set model path with:

```bash
export ICUB_MUJOCO_MODEL_PATH=/home/icub/mujoco_ws/icub-lerobot/model/xml/scene.xml
```

- Optional shared config:

```bash
export ICUB_LEROBOT_CONFIG=/home/icub/mujoco_ws/icub-lerobot/utils/control_config.yaml
```
