# lerobot_robot_icubrg

LeRobot robot plugin for iCub (Real/Gazebo) through YARP. Works against **both** the real
robot and the Gazebo simulation — the only difference is the `robot_name`
field (`icub` vs `icubSim`) in `control_config.yaml`, which selects the
camera source ports and is read by `../yarp_module/icubyarpinterface.py` on
the server side.

## Registered robot types

- `lerobot_robot_icubrg`
- `icub_remote`

## What it implements

- `RobotConfig` subclass: `iCubConfig`
- `Robot` subclass: `iCub`
- observation/action feature contract
- `connect`, `disconnect`, `get_observation`, `send_action`

This robot is a **YARP client**: it does not talk to the control boards or
the Cartesian solver directly. It connects to the ports opened by
`../yarp_module/teleop_module_sm.py`, which must already be running
(against the real robot or against Gazebo). See the top-level README for
the full startup sequence.

## Install

```bash
pip install .
```

## Runtime notes

- Requires YARP runtime and Python bindings (`import yarp`) in the same environment.
- Defaults are compatible with `icub-lerobot-gazebo/common/yarp_module/teleop_module_sm.py` ports.
- Set config path with:

```bash
export ICUB_LEROBOT_CONFIG=/home/icub/mujoco_ws/REPO_ICUB/icub-lerobot-gazebo/config/control_config.yaml
```

## Observation features

Unlike `lerobot_robot_icub_mujoco` (the MuJoCo robot used by `icub-lerobot-mj`), this robot
**does not** publish `object_pos_*` / `object_quat_*` observation features:
a real or Gazebo-simulated robot has no ground-truth pose for the
manipulated object (no motion-capture/vision system wired up), so those
fields simply don't exist here. Everything else — joint state, optional
joint velocity/torque/wrench/EEF-pose/tactile, and cameras — mirrors the
MuJoCo robot's feature set.
