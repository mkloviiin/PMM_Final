# MuJoCo 4 upgrade test (branch `test/mujoco-4`)

MuJoCo 3.6.0 (conda-forge) → **4.0.1 (PyPI)**. conda-forge has no 4.x build yet (latest 3.16.0),
so `mujoco` moved from `[dependencies]` to `[pypi-dependencies]` in `pixi.toml`. No source code
changes were needed: the project only uses `MjModel.from_xml_path`, `mj_step`/`mj_forward`,
`mj_resetDataKeyframe`, `mujoco.Renderer` and `mujoco.viewer.launch_passive`, all unchanged.

## Results (3.6.0 vs 4.0.1, same machine)

| Check | 3.6.0 | 4.0.1 |
|---|---|---|
| 6 scenes × {ik, id} MJCF: load, 3 s hold, NaN / warnings | ok | ok, same dims, objects within 0.3 mm |
| Eye cameras (`head_cam`, `head_cam_right`) | — | pixel-identical |
| `frontview` | — | mean abs diff 0.04–0.35 / 255 |
| `topview` | — | floor texture tiling shifted (~9 % px), see below |
| `replay_mujoco.py`, 5 eps each, lift / stack / pick-and-place (TSIK datasets of 2026-11-09) | 100 % success, EE RMSE 2.06 / 3.68 / 2.46 mm | 100 % success, EE RMSE 2.07 / 3.68 / 2.47 mm |
| Headless 6 cm circle tracking, lift scene, TSIK / TSID (EE RMSE) | 8.58 / 13.37 mm | 8.58 / 13.37 mm |
| Speed TSIK / TSID (× real time, headless) | 19.8 / 12.9 | 26.9 / 15.0 |
| Hub UI build, `mujoco.viewer.launch_passive` | ok | ok |

## Known differences

- **More contacts** per step (e.g. lift 9 → 18): `multiccd` is enabled by default since 3.8.
  Tracking and task outcomes did not change.
- **`topview` floor texture offset**: since 3.12 finite plane textures are anchored at the corner,
  not the center. Only the floor in the top camera; eye cameras (policy inputs) are unchanged.
- The segfault on interpreter exit after closing the passive viewer also happens on `main`
  (3.6.0, conda); it is not a regression.

## Fingers sinking into objects (hand fix)

Symptom: when the hand closes on the cube, the fingertips sink into it. Measured by replaying the
recorded TSIK datasets (10 episodes per task) and logging the deepest hand–object contact per step.

Cause: the finger position servos (kp 20–25 N·m/rad) were about 10× stiffer, in joint space, than
the contact with a light object. MuJoCo scales soft-contact stiffness by the effective mass, so a
28 g cube gives a soft contact. When the hand is told to close, the finger target lies inside the
object and the servo wins. It is not tunneling: cube speed stays < 0.4 m/s and the `speculative`
flag changes nothing.

Fix (`icub_visuomanip_{ikin,idyn}_limits.xml`, both hands):
- kp of the 18 finger actuators × 0.1.
- damping, stiffness and frictionloss of the 40 finger joints × 0.1. The same damping/kp ratio keeps
  the closing speed (free closing matches within 0.002 rad).
- `solimp="0.95 0.99 0.001 0.5 2"` on the hand collision meshes (`col_*`).

| Task | max penetration, mean / worst (old → new) | steps > 10 mm (old → new) | success |
|---|---|---|---|
| Block Lifting | 17.0 / 20.9 → 4.9 / 7.6 mm | 786 → 0 | 10/10 → 10/10 |
| Block Stacking | 13.6 / 20.7 → 3.4 / 5.1 mm | 569 → 0 | 10/10 → 10/10 |
| Pick-and-Place | 8.8 / 13.5 → 1.9 / 2.8 mm | 262 → 0 | 10/10 → 10/10 |

Rejected alternatives:
- Stiffer contacts only (solref/solimp, 2 ms timestep): penetration drops by at most about 40 %.
- Force limits on the finger actuators: fingers no longer close against their passive springs.
- Lower kp without lowering damping: fingers close about 10× slower and replays fail.
- Disabling `refsafe`: sustained penetration disappears, but contacts go unstable and the cube is
  ejected at up to 19 m/s.

Side effects:
- Grip force is about 10× lower. Every current object weighs ≤ 68 g, so this is still plenty
  (replays at 100 %).
- The finger `*_torque` observations are about 10× smaller than in older datasets.
- Door, Peg-In-Hole and Two-Arm Lifting load and simulate fine, but they have no replayable
  datasets (missing `meta/control_config.json`), so check them with teleoperation.

## Switching branches (pixi)

Going back from this branch to `main` leaves the conda `mujoco` package gutted (the PyPI
uninstall deletes files the conda reinstall just put in `site-packages/mujoco`). After checking
out `main`, run:

```bash
pixi install && pixi reinstall mujoco-python
```

Going from `main` to this branch only needs `pixi install`.
