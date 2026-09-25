# lerobot_teleoperator_icubrgteleop

LeRobot teleoperator plugin for iCub (Real/Gazebo) using MuJoCo mocap interaction.

## Registered teleoperator types

- `icubrg_teleop`
- `lerobot_teleoperator_icubrgteleop`

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

- In `icub-lerobot-gazebo`, this teleoperator is used purely as a **visualizer + IK
  solver**: it turns VR hand poses into Cartesian targets and mirrors robot feedback in
  a MuJoCo window, but the targets are actually sent to the real/Gazebo robot by
  `lerobot_robot_icubrg.iCub` (YARP), not to this MuJoCo model.
- Set model path with:

```bash
export ICUB_MUJOCO_MODEL_PATH=/home/icub/mujoco_ws/REPO_ICUB/icub-lerobot-gazebo/mujoco/assets/scenes/scene_icub_empty_table.xml
```

- Optional shared config:

```bash
export ICUB_LEROBOT_CONFIG=/home/icub/mujoco_ws/REPO_ICUB/icub-lerobot-gazebo/config/control_config.yaml
```
