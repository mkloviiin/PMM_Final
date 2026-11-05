#!/usr/bin/env python3
"""Deterministic replay of recorded LeRobot manipulation datasets in MuJoCo for iCub.

Supports loading datasets either from a local folder (--root) or downloading them directly from
Hugging Face Hub (--repo-id + --download-dir), restoring each episode's exact initial tabletop
object poses (converting from the recorded iCub `root` frame back to MuJoCo `world` coordinates),
executing the recorded actions through TSIK (`ik`) or TSID (`id` / `--impedance`), and measuring
both task success (`metrics/task_success.py`) and end-effector trajectory tracking error.

Examples:
  # 1. Download a dataset from Hugging Face into a chosen folder and replay 5 episodes:
  pixi run python replay_mujoco.py \\
      --repo-id CAMADA-UTFSM/BlockStacking_wb_icub \\
      --download-dir data/CAMADA-UTFSM_BlockStacking_wb_icub \\
      --scene "Block Stacking" --episodes 5

  # 2. Replay specific episodes from an already-local dataset:
  pixi run python replay_mujoco.py \\
      --root data/CAMADA-UTFSM_BlockStacking_wb_icub \\
      --episode 0 3 7

  # 3. Fast headless reproducibility check across all episodes:
  pixi run python replay_mujoco.py \\
      --root data/CAMADA-UTFSM_BlockStacking_wb_icub \\
      --episodes 0 --no-viewer --fast --out outputs/replay/block_stacking_check.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent

for plugin in [
    "src/plugins/lerobot-robot-icub",
    "src/plugins/lerobot-teleoperator-icubteleop",
    "src/plugins/lerobot-robot-icubrg",
    "src/plugins/lerobot-teleoperator-icubrgteleop",
]:
    p = repo_root / plugin
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

for p in (repo_root, repo_root / "src", repo_root / "metrics"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig  # noqa: E402
from lerobot_robot_icub.icub_mujoco import iCubMuJoCo  # noqa: E402

from eval_policy import MujocoSceneState, arms_of  # noqa: E402
from play import _load_scene_entries, _resolve_scene  # noqa: E402
import task_success  # noqa: E402
from tools.door_scene import place_door, set_handle_side  # noqa: E402
from tools.scene_reset import has_door, scene_entry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded LeRobot datasets (local or Hugging Face) on the MuJoCo iCub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo ID (e.g. CAMADA-UTFSM/BlockStacking_wb_icub) or local/* identifier.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset folder (containing meta/info.json). If --repo-id is also given and --root "
             "does not exist yet, the dataset is downloaded into --root.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Target directory where --repo-id should be downloaded from Hugging Face "
             "(default when --repo-id is remote and --root is omitted: data/<org>_<dataset>).",
    )
    parser.add_argument(
        "--use-hf-cache",
        action="store_true",
        help="Download --repo-id into the standard Hugging Face cache instead of materializing under data/.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download dataset files from Hugging Face even if the target directory already exists.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip downloading MP4 camera videos from Hugging Face (only meta/ and data/ are needed for MuJoCo replay).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private datasets (otherwise uses HF_TOKEN or cached `hf auth login`).",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Optional uncurated dataset folder to read initial obj{i}_pos/quat from if the curated dataset dropped them.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene index or label from scenes.yaml (e.g. 'Block Stacking', 'Door Opening'). "
             "Auto-inferred from dataset task/name when omitted.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional explicit path to a MuJoCo scene XML (overrides --scene XML selection).",
    )
    parser.add_argument(
        "--controller",
        choices=["ik", "id"],
        default="ik",
        help="Whole-body controller: 'ik' (TSIK kinematics) or 'id' (TSID dynamics).",
    )
    parser.add_argument(
        "--impedance",
        action="store_true",
        help="Enable compliant Cartesian impedance mode when --controller id.",
    )
    parser.add_argument(
        "--control-arms",
        choices=["right", "left", "both"],
        default=None,
        help="Active arm(s) (default: inferred automatically from the dataset's action features).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to replay starting from 0 (0 = replay all episodes in the dataset).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=None,
        help="Specific 0-based episode index/indices to replay (e.g. --episode 0 4 9). Overrides --episodes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Playback rate in Hz (default: dataset's recorded FPS from meta/info.json).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Step simulation as fast as possible without real-time sleep between frames.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless without opening the interactive MuJoCo viewer.",
    )
    parser.add_argument(
        "--door-side",
        choices=["left", "right", "auto"],
        default="auto",
        help="Door Opening scene: handle side ('auto' infers side from the active arm in the recorded episode).",
    )
    parser.add_argument(
        "--door-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Door Opening scene: explicit world (X, Y) center of the door panel.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write a JSON reproducibility report.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root / "config" / "control_config.yaml"),
        help="Path to control_config.yaml.",
    )
    return parser.parse_args()


def resolve_or_download_dataset(args: argparse.Namespace) -> Path:
    """Resolve a local dataset directory or download `--repo-id` from Hugging Face Hub."""
    target_dir_str = args.download_dir or args.root

    # Case 1: Explicit local path that already has meta/info.json (and no --force-download)
    if target_dir_str and not args.force_download:
        candidate = Path(target_dir_str).expanduser().resolve()
        if (candidate / "meta" / "info.json").is_file():
            return candidate
        if args.repo_id is None:
            raise FileNotFoundError(
                f"Local dataset not found at '{candidate}' (missing meta/info.json). "
                "Provide --repo-id <org/dataset> to download it from Hugging Face."
            )

    if not args.repo_id:
        raise SystemExit(
            "Specify either --root <local_dataset_dir> or --repo-id <org/dataset> "
            "(e.g. --repo-id CAMADA-UTFSM/BlockStacking_wb_icub)."
        )

    # Case 2: local/* pattern lookup under data/ and data_curated/
    if args.repo_id.startswith("local/"):
        pattern = f"{args.repo_id.replace('/', '_')}_*"
        candidates: list[Path] = []
        for folder in (repo_root / "data", repo_root / "data_curated", repo_root.parent / "data"):
            if folder.is_dir():
                candidates.extend(
                    p for p in folder.glob(pattern) if (p / "meta" / "info.json").is_file()
                )
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0].resolve()
        raise FileNotFoundError(
            f"No local dataset matching '{args.repo_id}' found under data/ or data_curated/."
        )

    # Case 3: Remote Hugging Face dataset repo_id
    from huggingface_hub import snapshot_download

    token = args.hf_token or os.environ.get("HF_TOKEN") or True
    ignore_patterns = ["videos/**"] if args.skip_videos else None

    if args.use_hf_cache and not target_dir_str:
        print(f"[replay] Downloading / resolving '{args.repo_id}' in Hugging Face cache...")
        cached_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
            force_download=args.force_download,
            ignore_patterns=ignore_patterns,
        )
        return Path(cached_dir).resolve()

    dest = (
        Path(target_dir_str).expanduser().resolve()
        if target_dir_str
        else (repo_root / "data" / args.repo_id.replace("/", "_")).resolve()
    )
    if (dest / "meta" / "info.json").is_file() and not args.force_download:
        print(f"[replay] Using existing downloaded dataset at '{dest}'")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[replay] Downloading dataset '{args.repo_id}' from Hugging Face -> '{dest}'...")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
        force_download=args.force_download,
        ignore_patterns=ignore_patterns,
    )
    if not (dest / "meta" / "info.json").is_file():
        raise RuntimeError(f"Downloaded repository '{args.repo_id}' does not contain meta/info.json at '{dest}'.")
    return dest


class ParquetReplayDataset:
    """Fast, zero-video-overhead reader for LeRobot v2/v3 Parquet datasets."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
        self.info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        self.fps: int = int(round(float(self.info.get("fps", 30))))
        self.features: dict[str, Any] = self.info.get("features", {})
        self.action_names: list[str] = list(self.features.get("action", {}).get("names") or [])
        self.state_names: list[str] = list(self.features.get("observation.state", {}).get("names") or [])

        ep_files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if ep_files:
            self.episodes_meta = (
                pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
                .sort_values("episode_index")
                .reset_index(drop=True)
            )
        else:
            self.episodes_meta = None

        self._cached_chunk_key: tuple[int, int] | None = None
        self._cached_chunk_df: pd.DataFrame | None = None
        self._all_data_df: pd.DataFrame | None = None

    @property
    def total_episodes(self) -> int:
        if self.episodes_meta is not None:
            return len(self.episodes_meta)
        return int(self.info.get("total_episodes", 0))

    def episode_indices(self) -> list[int]:
        if self.episodes_meta is not None and "episode_index" in self.episodes_meta.columns:
            return [int(x) for x in self.episodes_meta["episode_index"].tolist()]
        return list(range(self.total_episodes))

    def default_task_prompt(self) -> str:
        tasks_parquet = self.root / "meta" / "tasks.parquet"
        if tasks_parquet.is_file():
            try:
                tdf = pd.read_parquet(tasks_parquet)
                if len(tdf) > 0:
                    if "task" in tdf.columns:
                        return str(tdf["task"].iloc[0])
                    return str(tdf.index[0])
            except Exception:  # noqa: BLE001
                pass
        if self.episodes_meta is not None and "tasks" in self.episodes_meta.columns and len(self.episodes_meta) > 0:
            raw = self.episodes_meta["tasks"].iloc[0]
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            if isinstance(raw, (list, tuple)) and raw:
                return str(raw[0])
        return ""

    def load_episode_df(self, episode_index: int) -> pd.DataFrame:
        if self.episodes_meta is not None:
            matches = self.episodes_meta[self.episodes_meta["episode_index"] == episode_index]
            if not matches.empty:
                ep_row = matches.iloc[0]
                if "data/chunk_index" in ep_row and "data/file_index" in ep_row:
                    key = (int(ep_row["data/chunk_index"]), int(ep_row["data/file_index"]))
                    if self._cached_chunk_key != key or self._cached_chunk_df is None:
                        pq_path = (
                            self.root
                            / "data"
                            / f"chunk-{key[0]:03d}"
                            / f"file-{key[1]:03d}.parquet"
                        )
                        self._cached_chunk_df = pd.read_parquet(pq_path)
                        self._cached_chunk_key = key
                    df = self._cached_chunk_df
                    sub = df[df["episode_index"] == episode_index]
                    sort_col = "frame_index" if "frame_index" in sub.columns else "timestamp"
                    return sub.sort_values(sort_col).reset_index(drop=True)

        # Fallback for LeRobot v2 per-episode parquet files or generic data/*.parquet
        v2_candidates = sorted((self.root / "data").rglob(f"*episode_{episode_index:06d}.parquet"))
        if v2_candidates:
            sub = pd.read_parquet(v2_candidates[0])
            sort_col = "frame_index" if "frame_index" in sub.columns else "timestamp"
            return sub.sort_values(sort_col).reset_index(drop=True)

        if self._all_data_df is None:
            all_pq = sorted((self.root / "data").rglob("*.parquet"))
            if not all_pq:
                raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
            self._all_data_df = pd.concat([pd.read_parquet(p) for p in all_pq], ignore_index=True)
        sub = self._all_data_df[self._all_data_df["episode_index"] == episode_index]
        if sub.empty:
            raise KeyError(f"Episode index {episode_index} not found in dataset at {self.root}")
        sort_col = "frame_index" if "frame_index" in sub.columns else "timestamp"
        return sub.sort_values(sort_col).reset_index(drop=True)


def infer_scene_from_dataset(ds: ParquetReplayDataset, repo_id: str | None) -> str | None:
    """Infer the scene label in `scenes.yaml` from the recorded task string or dataset name."""
    entries = _load_scene_entries()
    prompt = ds.default_task_prompt().strip().lower()
    if prompt:
        for entry in entries:
            if str(entry.get("task", "")).strip().lower() == prompt:
                return str(entry["label"])

    hint = f"{repo_id or ''} {ds.root.name} {prompt}".lower()
    keyword_map = [
        (("stack", "blockstacking"), "Block Stacking"),
        (("door", "dooropening"), "Door Opening"),
        (("pickplace", "pickandplace", "pick_and_place", "pick-and-place"), "Pick-and-Place"),
        (("peginhole", "peg_in_hole", "peg-in-hole"), "Two-Arm Peg-In-Hole"),
        (("twoarmlift", "two_arm_lift", "two-arm-lift", "lift-pot"), "Two-Arm Lifting"),
        (("blocklifting", "block_lifting", "block-lifting", "blue block", "lift"), "Block Lifting"),
    ]
    for keywords, label in keyword_map:
        if any(k in hint for k in keywords):
            return label
    return None


def _infer_door_side_from_actions(actions_2d: np.ndarray, action_idx: dict[str, int]) -> str:
    """Infer which handle side ('right' or 'left') was active from the recorded hand trajectories."""
    def _arm_span(prefix: str) -> float:
        cols = [action_idx[f"{prefix}_pos_{ax}"] for ax in "xyz" if f"{prefix}_pos_{ax}" in action_idx]
        if len(cols) != 3 or len(actions_2d) < 2:
            return 0.0
        traj = actions_2d[:, cols]
        return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())

    rh_span = _arm_span("rh")
    lh_span = _arm_span("lh")
    return "left" if lh_span > rh_span * 1.15 else "right"


def reset_and_restore_episode(
    robot: iCubMuJoCo,
    scene_objects: list[dict[str, Any]],
    scene_joints: list[dict[str, Any]],
    model_path: Path,
    first_state: np.ndarray | None,
    state_idx: dict[str, int],
    actions_2d: np.ndarray,
    action_idx: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reset robot to home posture, settle for 0.5 s, and restore recorded frame-0 object/door poses."""
    tc = robot.teleop_core
    model, data = tc.model, tc.data

    # 1. Reset robot to home with open hands and settle for 0.5 s (matching play.py / eval_policy.py)
    data.qpos[:] = tc.home_qpos
    data.ctrl[:] = tc.home_ctrl
    tc._apply_home_pose_to_arrays(data.qpos, data.ctrl)
    data.qvel[:] = 0.0
    tc._set_hand_level("right", 0.0)
    tc._set_hand_level("left", 0.0)
    mujoco.mj_forward(model, data)
    tc._snap_targets_to_ee()
    tc.state = "teleop_active"

    for _ in range(int(0.5 / tc.control_dt)):
        tc._solve_whole_body()
        tc._interpolate_hands()
        tc._step_physics()
    tc._snap_targets_to_ee()

    # 2. Reset scene joints (e.g. door_hinge = 0, handle_hinge = 0)
    for jnt_def in scene_joints or []:
        jname = jnt_def.get("name", "")
        try:
            jid = model.joint(jname).id
            data.qpos[model.jnt_qposadr[jid]] = float(jnt_def.get("qpos", 0.0))
            data.qvel[model.jnt_dofadr[jid]] = 0.0
        except KeyError:
            continue

    restored_info: dict[str, Any] = {"restored_objects": {}}

    # 3. Handle Door Opening configuration if applicable
    if has_door(model):
        side = (
            _infer_door_side_from_actions(actions_2d, action_idx)
            if args.door_side == "auto"
            else args.door_side
        )
        set_handle_side(model, side)
        mujoco.mj_forward(model, data)
        door_xy = args.door_xy
        if door_xy is None:
            ent = scene_entry(model_path)
            if ent.get("door") and ent["door"].get("panel_xy"):
                door_xy = ent["door"]["panel_xy"]
        if door_xy is not None:
            place_door(model, data, door_xy)
            mujoco.mj_forward(model, data)
        restored_info["door_side"] = side
        if door_xy is not None:
            restored_info["door_xy"] = [round(float(door_xy[0]), 3), round(float(door_xy[1]), 3)]

    # 4. Restore tabletop object poses from recorded frame-0 state (`root` -> `world`)
    if first_state is not None and scene_objects:
        for i, obj_def in enumerate(scene_objects):
            body_name = obj_def.get("body", "")
            prefix = f"obj{i}"
            pos_keys = [f"{prefix}_pos_{ax}" for ax in "xyz"]
            quat_keys = [f"{prefix}_quat_{ax}" for ax in "wxyz"]

            # Legacy single-object fallback (`object_pos_*` / `object_quat_*`)
            if i == 0 and not all(k in state_idx for k in pos_keys + quat_keys):
                pos_keys = [f"object_pos_{ax}" for ax in "xyz"]
                quat_keys = [f"object_quat_{ax}" for ax in "wxyz"]
                prefix = "object"

            if not all(k in state_idx for k in pos_keys + quat_keys):
                continue

            raw_pos = np.array([float(first_state[state_idx[k]]) for k in pos_keys], dtype=np.float64)
            raw_quat = np.array([float(first_state[state_idx[k]]) for k in quat_keys], dtype=np.float64)
            if np.linalg.norm(raw_quat) < 1e-4 and np.linalg.norm(raw_pos) < 1e-4:
                continue

            if robot.config.pose_frame == "root":
                conv = {
                    **{f"{prefix}_pos_{ax}": raw_pos[j] for j, ax in enumerate("xyz")},
                    **{f"{prefix}_quat_{ax}": raw_quat[j] for j, ax in enumerate("wxyz")},
                }
                robot._convert_poses(conv, [(f"{prefix}_pos", f"{prefix}_quat")], to_root=False)
                world_pos = np.array([float(conv[f"{prefix}_pos_{ax}"]) for ax in "xyz"], dtype=np.float64)
                world_quat = np.array([float(conv[f"{prefix}_quat_{ax}"]) for ax in "wxyz"], dtype=np.float64)
            else:
                world_pos = raw_pos
                world_quat = raw_quat

            qnorm = np.linalg.norm(world_quat)
            world_quat = (world_quat / qnorm) if qnorm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])

            try:
                bid = model.body(body_name).id
                jnt = int(model.body_jntadr[bid])
                if jnt >= 0:
                    qadr = int(model.jnt_qposadr[jnt])
                    dadr = int(model.jnt_dofadr[jnt])
                    data.qpos[qadr : qadr + 3] = world_pos
                    data.qpos[qadr + 3 : qadr + 7] = world_quat
                    ndof = 6 if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE else 1
                    data.qvel[dadr : dadr + ndof] = 0.0
                else:
                    model.body_pos[bid] = world_pos
                    model.body_quat[bid] = world_quat
                restored_info["restored_objects"][body_name] = np.round(world_pos, 4).tolist()
            except KeyError:
                continue

        mujoco.mj_forward(model, data)

    if scene_objects and not restored_info["restored_objects"]:
        print(
            "[replay][warn] Dataset observation.state does not contain obj{i}_pos/quat features; "
            "falling back to random scene spawn. Pass --raw-root <uncurated_dataset> for exact initial poses."
        )
        tc._reset_scenario()

    return restored_info


def main() -> None:
    args = parse_args()
    if args.impedance and args.controller != "id":
        print("[replay] --impedance only applies to --controller id: ignored.")
        args.impedance = False

    dataset_dir = resolve_or_download_dataset(args)
    ds = ParquetReplayDataset(dataset_dir)
    raw_ds = ParquetReplayDataset(Path(args.raw_root).expanduser()) if args.raw_root else None

    fps = args.fps or ds.fps or 30
    scene_query = args.scene or infer_scene_from_dataset(ds, args.repo_id)
    model_path, default_task, scene_objects, scene_joints, cameras = _resolve_scene(
        scene_query, controller=args.controller, model_file=args.model
    )
    success_task = task_success.TASK_BY_SCENE.get(model_path.name)

    control_arms = args.control_arms or arms_of(ds.action_names) or "both"
    config_path = str(Path(args.config).expanduser().resolve())

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=config_path,
        control_arms=control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
        control_dt=1.0 / fps,
        vr_enabled=False,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
        camera_names=[],  # No offscreen camera rendering needed during open-loop action replay
    )
    robot = iCubMuJoCo(robot_cfg)
    robot.connect()

    action_idx = {name: i for i, name in enumerate(ds.action_names)}
    state_source = raw_ds if raw_ds is not None else ds
    state_idx = {name: i for i, name in enumerate(state_source.state_names)}

    robot_action_keys = list(robot.action_features.keys())
    missing_actions = [k for k in robot_action_keys if k not in action_idx]
    if missing_actions:
        print(f"[replay][warn] Action keys missing in dataset (set to 0.0): {missing_actions}")

    all_ep_indices = ds.episode_indices()
    if args.episode is not None:
        selected_eps = [ep for ep in args.episode if ep in all_ep_indices]
        missing_eps = [ep for ep in args.episode if ep not in all_ep_indices]
        if missing_eps:
            print(f"[replay][warn] Episode indices not in dataset (skipped): {missing_eps}")
    elif args.episodes <= 0:
        selected_eps = all_ep_indices
    else:
        selected_eps = all_ep_indices[: args.episodes]

    print("\n=======================================================")
    print(" Starting Dataset Replay (MuJoCo)")
    print(f" Dataset:    {dataset_dir} ({ds.total_episodes} total episodes)")
    print(f" Scene:      {model_path.name}  |  Task: {ds.default_task_prompt() or default_task}")
    print(f" Controller: {args.controller.upper()} {'(Impedance)' if args.impedance else ''}")
    print(f" Arms:       {control_arms}  |  FPS: {fps} Hz  |  Episodes to replay: {len(selected_eps)}")
    if success_task:
        print(f" Criterion:  {success_task}: {task_success.DESCRIPTION[success_task]}")
    print("=======================================================\n")

    tc = robot.teleop_core
    scene_state = MujocoSceneState(tc.model, success_task) if success_task else None

    mj_viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data, show_left_ui=False, show_right_ui=False)
        frontview = mujoco.mj_name2id(tc.model, mujoco.mjtObj.mjOBJ_CAMERA, "frontview")
        if frontview >= 0:
            with mj_viewer.lock():
                mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                mj_viewer.cam.fixedcamid = frontview
        else:
            mj_viewer.cam.azimuth, mj_viewer.cam.elevation, mj_viewer.cam.distance = 135, -20, 2.0

    ee_check_sides = [
        s for s in ("rh", "lh")
        if s in robot._arm_sides() and all(f"{s}_ee_pos_{ax}" in state_idx for ax in "xyz")
    ]

    results: list[dict[str, Any]] = []
    try:
        for run_i, ep_idx in enumerate(selected_eps, start=1):
            if mj_viewer is not None and not mj_viewer.is_running():
                print("[replay] Viewer closed by user -> stopping.")
                break

            ep_df = ds.load_episode_df(ep_idx)
            actions_2d = np.stack(ep_df["action"].to_numpy())
            if actions_2d.ndim == 1:
                actions_2d = actions_2d[:, None]

            states_2d: np.ndarray | None = None
            if raw_ds is not None:
                raw_ep_df = raw_ds.load_episode_df(ep_idx)
                if "observation.state" in raw_ep_df.columns and len(raw_ep_df) > 0:
                    states_2d = np.stack(raw_ep_df["observation.state"].to_numpy())
            elif "observation.state" in ep_df.columns and len(ep_df) > 0:
                states_2d = np.stack(ep_df["observation.state"].to_numpy())

            first_state = states_2d[0] if states_2d is not None and len(states_2d) > 0 else None
            restored = reset_and_restore_episode(
                robot=robot,
                scene_objects=scene_objects,
                scene_joints=scene_joints,
                model_path=model_path,
                first_state=first_state,
                state_idx=state_idx,
                actions_2d=actions_2d,
                action_idx=action_idx,
                args=args,
            )
            if mj_viewer is not None:
                mj_viewer.sync()

            ep_success = task_success.EpisodeSuccess(success_task) if scene_state is not None else None
            ee_sq_errors: list[float] = []
            num_frames = len(actions_2d)

            for f in range(num_frames):
                if mj_viewer is not None and not mj_viewer.is_running():
                    break
                t0 = time.perf_counter()

                # Compare pre-action measured EE pose against recorded frame f state (if available)
                if states_2d is not None and f < len(states_2d) and ee_check_sides:
                    obs_now = robot.get_observation()
                    for side in ee_check_sides:
                        rec_p = np.array([states_2d[f, state_idx[f"{side}_ee_pos_{ax}"]] for ax in "xyz"])
                        sim_p = np.array([float(obs_now[f"{side}_ee_pos_{ax}"]) for ax in "xyz"])
                        ee_sq_errors.append(float(np.sum((sim_p - rec_p) ** 2)))

                if ep_success is not None:
                    ep_success.update(f, scene_state(tc.data))

                act_vec = actions_2d[f]
                action_dict = {
                    key: float(act_vec[action_idx[key]]) if key in action_idx else 0.0
                    for key in robot_action_keys
                }
                robot.send_action(action_dict)

                if mj_viewer is not None:
                    mj_viewer.sync()
                if not args.fast:
                    elapsed = time.perf_counter() - t0
                    precise_sleep(max(1.0 / fps - elapsed, 0.0))

            if ep_success is not None:
                ep_success.update(num_frames, scene_state(tc.data))
                metrics = ep_success.summary()
            else:
                metrics = {"success": None, "success_step": None, "success_at_end": None}

            ee_rmse_mm = round(float(np.sqrt(np.mean(ee_sq_errors)) * 1000.0), 2) if ee_sq_errors else None
            ep_report = {
                "episode": int(ep_idx),
                "frames": num_frames,
                **restored,
                **metrics,
                "ee_rmse_mm": ee_rmse_mm,
            }
            results.append(ep_report)

            objs_str = ", ".join(f"{k}={v}" for k, v in restored.get("restored_objects", {}).items())
            rmse_str = f" | EE RMSE={ee_rmse_mm:.2f} mm" if ee_rmse_mm is not None else ""
            print(
                f"[{run_i}/{len(selected_eps)}] Episode {ep_idx:3d} ({num_frames} frames): "
                f"success={metrics['success']} (at_end={metrics['success_at_end']}){rmse_str}"
                + (f" | init: {objs_str}" if objs_str else "")
            )

        measured = [r for r in results if r["success"] is not None]
        summary = {
            "dataset": str(dataset_dir),
            "repo_id": args.repo_id,
            "scene": model_path.name,
            "controller": args.controller,
            "impedance": args.impedance,
            "episodes_replayed": len(results),
            "reproducibility_success_rate": (
                round(sum(bool(r["success"]) for r in measured) / len(measured), 4) if measured else None
            ),
            "reproducibility_at_end_rate": (
                round(sum(bool(r["success_at_end"]) for r in measured) / len(measured), 4) if measured else None
            ),
            "mean_ee_rmse_mm": (
                round(float(np.mean([r["ee_rmse_mm"] for r in results if r["ee_rmse_mm"] is not None])), 2)
                if any(r["ee_rmse_mm"] is not None for r in results)
                else None
            ),
            "episodes": results,
        }
        if measured:
            print(
                f"\n[replay] Reproducibility summary ({len(measured)} episodes): "
                f"success={summary['reproducibility_success_rate']:.1%}  "
                f"at_end={summary['reproducibility_at_end_rate']:.1%}"
                + (f"  mean_EE_RMSE={summary['mean_ee_rmse_mm']} mm" if summary["mean_ee_rmse_mm"] is not None else "")
            )
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[replay] Saved report to {out_path}")
    finally:
        if mj_viewer is not None:
            mj_viewer.close()
        if robot.is_connected:
            robot.disconnect()
        print("[replay] Finished.")


if __name__ == "__main__":
    main()
