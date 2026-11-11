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

## Switching branches (pixi)

Going back from this branch to `main` leaves the conda `mujoco` package gutted (the PyPI
uninstall deletes files the conda reinstall just put in `site-packages/mujoco`). After checking
out `main`, run:

```bash
pixi install && pixi reinstall mujoco-python
```

Going from `main` to this branch only needs `pixi install`.
