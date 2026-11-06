#!/usr/bin/env python3
"""Unified policy evaluation for trained LeRobot VLA models across MuJoCo, Gazebo, and Real iCub.

Supported Backends:
  - mujoco: Evaluates policy inside MuJoCo physics with TSIK (ik) or TSID (id).
  - gazebo: Evaluates policy against Gazebo Sim via YARP / teleop_module_sm.
  - real: Evaluates policy against the Real iCub robot via YARP / teleop_module_sm.

Success: robosuite's _check_success of the task (metrics/task_success.py), the same code for
every simulator: success if it holds at some step (as robomimic rollouts) and also at the end
of the episode. Block Stacking uses its own criterion, only on the final state, and reports the
success rate per side of the blue cube (MuJoCo alternates the mirrored layout every episode). MuJoCo builds the scene state from MjData; Gazebo from its topics (object poses,
fingertip contact sensors, door joint states); on the real robot the operator answers.

Each episode starts at home with the hands open and the scene reset (scenes.yaml ranges; in
MuJoCo, tools/scene_reset.py: the side of the task alternates every episode).
Results: outputs/eval/<checkpoint>_<backend>_<date>/:
  eval.log, summary.json (success rate total / at the end / cumulative / per side, per-episode
  metrics), videos/<camera>.mp4 (one video per policy camera with every episode in a row; each
  episode's start / end in summary.json)

    pixi run python eval_policy.py --checkpoint outputs/train/.../pretrained_model --scene "Block Lifting"
    pixi run python eval_policy.py --checkpoint ... --backend gazebo --scene 3 --controller id --impedance
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

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
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402

import task_success  # noqa: E402
from play import _resolve_scene  # noqa: E402

# Gazebo: time for go_home + reset_objects.sh before the episode starts [s]
GAZEBO_RESET_WAIT_S = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trained LeRobot Policy on iCub")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local folder path or Hugging Face model repo ID for the trained policy checkpoint "
             "(e.g. output/train/<run>/checkpoints/last/pretrained_model or CAMADA-UTFSM/<model_repo>)",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional local directory where a Hugging Face --checkpoint repo ID should be downloaded "
             "(e.g. models/CAMADA-UTFSM_smolvla_stacking)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download model files from Hugging Face when --download-dir is used",
    )
    parser.add_argument(
        "--dataset-meta",
        default=None,
        help="Optional local dataset directory or Hugging Face dataset repo ID "
             "(e.g. CAMADA-UTFSM/BlockStacking_wb_icub) to load exact observation.state and action "
             "feature names when evaluating a checkpoint trained on another machine",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for private models/datasets (falls back to HF_TOKEN or `hf auth login`)",
    )
    parser.add_argument(
        "--backend",
        choices=["mujoco", "gazebo", "real"],
        default="mujoco",
        help="Evaluation backend",
    )
    parser.add_argument(
        "--controller",
        choices=["ik", "id"],
        default="ik",
        help="Whole-body controller type: 'ik' (TSIK) or 'id' (TSID)",
    )
    parser.add_argument(
        "--impedance",
        action="store_true",
        help="Enable compliant Cartesian impedance mode when --controller id",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--episode-len", type=int, default=600, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=30, help="Control rate (as recorded)")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="Actions executed from each predicted chunk before predicting again "
                             "(default: the policy's n_action_steps). Fewer = more closed-loop corrections")
    parser.add_argument("--task", default=None, help="Language task instruction prompt (default: scenes.yaml)")
    parser.add_argument("--scene", default=None, help="Scene index or label in scenes.yaml (e.g. 1 or 'Door Opening')")
    parser.add_argument("--control-arms", choices=["right", "left", "both"], default=None,
                        help="Active arms (default: the arms in the policy's actions)")
    parser.add_argument("--robot-name", choices=["icub", "icubSim"], default=None, help="YARP robot name")
    parser.add_argument("--no-viewer", action="store_true", help="Do not show the camera / MuJoCo windows")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the scene resets (MuJoCo)")
    parser.add_argument("--door-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Door scene (MuJoCo): center of the door panel at (X, Y) in the world")
    parser.add_argument("--door-side", choices=["left", "right", "alternate"], default=None,
                        help="Door scene (MuJoCo): handle side (default: --mirror, i.e. alternate). "
                             "The old scenario: --door-side left --door-xy 0.47 -0.13")
    parser.add_argument("--side", choices=["right", "left"], default=None,
                        help="MuJoCo: fixed side of the task (object to pick / door handle) in every "
                             "episode, e.g. for a policy trained on one side only")
    parser.add_argument("--mirror", choices=["alternate", "random"], default="alternate",
                        help="Two-sided scenes (scenes.yaml sides / mirror_y), MuJoCo: alternate = right "
                             "in odd episodes, left (mirrored) in even ones; random = 50/50 draw")
    parser.add_argument("--no-video", action="store_true",
                        help="Do not save the videos of the policy cameras (run folder videos/<camera>.mp4)")
    parser.add_argument("--out", default=None, help="Results folder (default: outputs/eval/<name>_<backend>_<date>)")
    parser.add_argument("--config", default=str(repo_root / "config" / "control_config.yaml"), help="Config path")
    return parser.parse_args()


class _Tee:
    """Copies stdout/stderr to the run's eval.log."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _load_dataset_info_features(
    dataset_ref: str | Path,
    token: str | bool | None = None,
) -> tuple[list[str] | None, list[str] | None]:
    """Read (state_names, action_names) from a local dataset folder or a Hugging Face dataset repo ID."""
    cand = Path(dataset_ref).expanduser()
    info_path = cand / "meta" / "info.json"
    if info_path.is_file():
        feats = json.loads(info_path.read_text(encoding="utf-8")).get("features", {})
        return (
            feats.get("observation.state", {}).get("names"),
            feats.get("action", {}).get("names"),
        )
    if isinstance(dataset_ref, str) and "/" in dataset_ref and not dataset_ref.startswith("local/") and not cand.exists():
        try:
            from huggingface_hub import hf_hub_download

            dl_info = Path(
                hf_hub_download(
                    repo_id=dataset_ref,
                    filename="meta/info.json",
                    repo_type="dataset",
                    token=token,
                )
            )
            feats = json.loads(dl_info.read_text(encoding="utf-8")).get("features", {})
            return (
                feats.get("observation.state", {}).get("names"),
                feats.get("action", {}).get("names"),
            )
        except Exception:  # noqa: BLE001
            return None, None
    return None, None


def resolve_checkpoint_source(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve `--checkpoint` (local folder or Hugging Face repo ID) into `(pretrained_source, run_name)`.

    When `--download-dir` is given alongside a Hugging Face repo ID, downloads the model snapshot into
    `--download-dir` first so subsequent evaluations run directly from disk.
    """
    local_ckpt = Path(args.checkpoint).expanduser()
    if local_ckpt.exists():
        resolved = local_ckpt.resolve()
        if resolved.is_dir() and not (resolved / "config.json").is_file() and (resolved / "pretrained_model" / "config.json").is_file():
            resolved = resolved / "pretrained_model"
        run_name = (
            resolved.parent.parent.name
            if resolved.is_dir() and resolved.name == "pretrained_model"
            else resolved.name
        )
        return str(resolved), run_name

    # Remote Hugging Face model repo_id (e.g. CAMADA-UTFSM/smolvla_block_stacking)
    repo_id = args.checkpoint.strip()
    run_name = repo_id.strip("/").replace("/", "_")
    if not args.download_dir:
        return repo_id, run_name

    from huggingface_hub import snapshot_download

    token = args.hf_token or os.environ.get("HF_TOKEN") or True
    dest = Path(args.download_dir).expanduser().resolve()
    has_model = (dest / "config.json").is_file() or (dest / "pretrained_model" / "config.json").is_file()
    if has_model and not args.force_download:
        print(f"[eval] Using existing downloaded model at '{dest}'")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[eval] Downloading model '{repo_id}' from Hugging Face -> '{dest}'...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(dest),
            token=token,
            force_download=args.force_download,
        )
    if (dest / "pretrained_model" / "config.json").is_file() and not (dest / "config.json").is_file():
        dest = dest / "pretrained_model"
    return str(dest), run_name


def training_feature_names(
    checkpoint: str | Path,
    dataset_meta: str | None = None,
    token: str | bool | None = None,
) -> tuple[list[str] | None, list[str] | None]:
    """Recover `(state_names, action_names)` of the training dataset across local and HF checkpoints."""
    # 1. Explicit --dataset-meta (local path or HF dataset repo_id)
    if dataset_meta:
        s_names, a_names = _load_dataset_info_features(dataset_meta, token=token)
        if s_names is not None or a_names is not None:
            return s_names, a_names

    # 2. Read train_config.json from local checkpoint dir or remote HF model repo
    cfg: dict | None = None
    ckpt_path = Path(checkpoint).expanduser()
    if ckpt_path.is_dir() and (ckpt_path / "train_config.json").is_file():
        try:
            cfg = json.loads((ckpt_path / "train_config.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = None
    elif isinstance(checkpoint, str) and "/" in checkpoint and not ckpt_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            tc_file = Path(
                hf_hub_download(
                    repo_id=checkpoint,
                    filename="train_config.json",
                    repo_type="model",
                    token=token,
                )
            )
            cfg = json.loads(tc_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            cfg = None

    if not cfg or not isinstance(cfg.get("dataset"), dict):
        return None, None

    ds_cfg = cfg["dataset"]
    # 2a. Try the exact training root path or matching folder name inside data/ or data_curated/
    ds_root = ds_cfg.get("root")
    if ds_root:
        root_p = Path(ds_root).expanduser()
        for candidate in (
            root_p,
            repo_root / "data" / root_p.name,
            repo_root / "data_curated" / root_p.name,
        ):
            s_names, a_names = _load_dataset_info_features(candidate, token=token)
            if s_names is not None or a_names is not None:
                return s_names, a_names

    # 2b. Try dataset repo_id (first under local data/ folders, then via HF Hub meta/info.json)
    ds_repo_id = ds_cfg.get("repo_id")
    if isinstance(ds_repo_id, str) and ds_repo_id:
        sanitized = ds_repo_id.replace("/", "_")
        for candidate in (repo_root / "data" / sanitized, repo_root / "data_curated" / sanitized):
            s_names, a_names = _load_dataset_info_features(candidate, token=token)
            if s_names is not None or a_names is not None:
                return s_names, a_names
        if not ds_repo_id.startswith("local/"):
            s_names, a_names = _load_dataset_info_features(ds_repo_id, token=token)
            if s_names is not None or a_names is not None:
                return s_names, a_names

    return None, None


def arms_of(action_names: list[str] | None) -> str | None:
    if not action_names:
        return None
    has_r = any(n.startswith("rh_") for n in action_names)
    has_l = any(n.startswith("lh_") for n in action_names)
    return "both" if has_r and has_l else "right" if has_r else "left" if has_l else None


def infer_arms_from_dim(policy_cfg) -> str:
    """Fallback when action feature names are unavailable: >=16 action dims implies bimanual ('both')."""
    try:
        act_feat = (getattr(policy_cfg, "output_features", None) or {}).get("action")
        shape = getattr(act_feat, "shape", None) or (act_feat.get("shape") if isinstance(act_feat, dict) else None)
        if shape and int(shape[0]) >= 16:
            return "both"
    except Exception:  # noqa: BLE001
        pass
    return "right"


def to_float(value) -> float:
    return float(value.item()) if isinstance(value, torch.Tensor) else float(value)


# ── Scene state for the success criterion ─────────────────────────────────────
def _body_geoms(model, body_names) -> set[int]:
    ids = {model.body(n).id for n in body_names}
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in ids}


def _in_contact(data, geoms_a: set[int], geoms_b: set[int]) -> bool:
    """Like robosuite's check_contact: any contact between the two geom groups."""
    for c in data.contact[:data.ncon]:
        if (c.geom1 in geoms_a and c.geom2 in geoms_b) or (c.geom2 in geoms_a and c.geom1 in geoms_b):
            return True
    return False


class MujocoSceneState:
    """task_success.SceneState from MuJoCo: poses, grasps (thumb tip and index or middle tip, like
    the Gazebo sensors), contacts between objects and joints."""

    def __init__(self, model, task: str):
        self.task = task
        self.bodies = {b: model.body(b).id for b in task_success.BODIES[task]}
        self.geoms = {b: _body_geoms(model, [b]) for b in self.bodies}
        self.table_site = model.site("table_top").id
        self.hands = [
            (_body_geoms(model, [f"{s}_hand_{task_success.GRASP_THUMB}_3"]),
             _body_geoms(model, [f"{s}_hand_{f}_3" for f in task_success.GRASP_FINGERS]))
            for s in ("r", "l")
        ]
        self.joints = {j: model.jnt_qposadr[model.joint(j).id] for j in task_success.JOINTS.get(task, [])}

    def __call__(self, data) -> task_success.SceneState:
        grasped = {b for b in task_success.GRASP_BODIES.get(self.task, [])
                   if any(_in_contact(data, thumb, self.geoms[b]) and _in_contact(data, fingers, self.geoms[b])
                          for thumb, fingers in self.hands)}
        touching = {frozenset(pair) for pair in task_success.CONTACT_PAIRS.get(self.task, [])
                    if _in_contact(data, self.geoms[pair[0]], self.geoms[pair[1]])}
        return task_success.SceneState(
            poses={b: (data.xpos[i].copy(), data.xquat[i].copy()) for b, i in self.bodies.items()},
            table_z=float(data.site_xpos[self.table_site][2]),
            grasped=grasped, touching=touching,
            joints={j: float(data.qpos[adr]) for j, adr in self.joints.items()},
        )


class GazeboSceneState:
    """task_success.SceneState from Gazebo: poses in the Gazebo world (PosePublisher of each
    model), grasps with the fingertip sensors (/icub/touch/<hand>_<finger>), contacts between
    objects with the model's sensor (/<body>/contact, e.g. models/red-cube) and joints with the
    JointStatePublisher (the door)."""

    HANDS = ("rh", "lh")

    def __init__(self, task: str):
        from lerobot_robot_icubrg.gz_pose import GzContactReader, GzJointReader, GzPoseReader

        self.task = task
        self.bodies = task_success.BODIES[task]
        self.grasp_bodies = task_success.GRASP_BODIES.get(task, [])
        self.pairs = task_success.CONTACT_PAIRS.get(task, [])
        self.pose_reader = GzPoseReader(self.bodies, (0.0, 0.0, 0.0))  # Gazebo world
        fingers = (task_success.GRASP_THUMB, *task_success.GRASP_FINGERS)
        keys = [f"{h}_{f}" for h in self.HANDS for f in fingers] if self.grasp_bodies else []
        topics = {b: f"/{b}/contact" for _, b in self.pairs}
        self.contacts = GzContactReader(keys + list(topics), topics=topics) if keys or topics else None
        self.joint_reader = GzJointReader("door") if task_success.JOINTS.get(task) else None

    def start(self) -> list[str]:
        """Starts the readers; returns the contact topics Gazebo does not advertise (yet)."""
        self.pose_reader.start()
        if self.joint_reader is not None:
            self.joint_reader.start()
        return self.contacts.start() if self.contacts is not None else []

    def stop(self) -> None:
        self.pose_reader.stop()
        if self.contacts is not None:
            self.contacts.stop()
        if self.joint_reader is not None:
            self.joint_reader.stop()

    def _grasped(self, body: str) -> bool:
        c = self.contacts
        return any(c.touching(f"{h}_{task_success.GRASP_THUMB}", body)
                   and any(c.touching(f"{h}_{f}", body) for f in task_success.GRASP_FINGERS)
                   for h in self.HANDS)

    def wait(self, since: float, timeout_s: float = 5.0) -> list[str]:
        """Waits for a pose newer than `since` of every object; returns the missing ones."""
        t0 = time.time()
        while time.time() < t0 + timeout_s:
            late = [b for b in self.bodies if (self.pose_reader.stamp(b) or 0.0) < since]
            if not late:
                return []
            time.sleep(0.05)
        return late

    def __call__(self, since: float) -> task_success.SceneState | None:
        """Current state, or None while an object has no pose newer than `since`
        (reset_objects.sh recreates the objects: the previous episode's pose is stale)."""
        if any((self.pose_reader.stamp(b) or 0.0) < since for b in self.bodies):
            return None
        return task_success.SceneState(
            poses={b: self.pose_reader.get(b) for b in self.bodies},
            table_z=task_success.TABLE_Z,
            grasped={b for b in self.grasp_bodies if self._grasped(b)},
            touching={frozenset((a, b)) for a, b in self.pairs if self.contacts.touching(b, a)},
            joints={j: self.joint_reader.get(j) or 0.0 for j in task_success.JOINTS.get(self.task, [])}
            if self.joint_reader is not None else {},
        )


def blue_side(s: task_success.SceneState, backend: str) -> str:
    """Side of the blue cube seen from the robot: MuJoCo y points to its left; the Gazebo world
    is turned 180 deg in z ((x, y) -> (-x, -y), reset_objects.sh)."""
    y = s.pos("blue-cube")[1] * (-1.0 if backend == "gazebo" else 1.0)
    return "left" if y > 0 else "right"


def rate(episodes: list[dict], key: str) -> float | None:
    return sum(bool(m[key]) for m in episodes) / len(episodes) if episodes else None


def pct(r: float | None) -> float | None:
    return None if r is None else round(100 * r, 1)


class CameraVideos:
    """One .mp4 per policy camera (what it sees every step) with every episode in a row:
    <run>/videos/<camera>.mp4. n_frames gives each episode's start / end."""

    def __init__(self, run_dir: Path, cameras: list[str], fps: int):
        self.run_dir, self.fps = run_dir, fps
        self.paths = {cam: run_dir / "videos" / f"{cam}.mp4" for cam in cameras}
        self.containers, self.streams = {}, {}  # opened with the first frame (its size)
        self.n_frames = 0

    @property
    def time_s(self) -> float:
        return round(self.n_frames / self.fps, 3)

    def files(self) -> dict[str, str]:
        return {cam: str(path.relative_to(self.run_dir)) for cam, path in self.paths.items()}

    def add(self, obs: dict) -> None:
        import av

        for cam, path in self.paths.items():
            img = obs.get(cam)
            if img is None:
                continue
            img = np.ascontiguousarray(img)
            img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]  # yuv420p: even sides
            if cam not in self.streams:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.containers[cam] = av.open(str(path), mode="w")
                stream = self.containers[cam].add_stream("libx264", rate=self.fps)
                stream.width, stream.height = img.shape[1], img.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "veryfast"}
                self.streams[cam] = stream
            for packet in self.streams[cam].encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                self.containers[cam].mux(packet)
        self.n_frames += 1

    def close(self) -> None:
        for cam, stream in self.streams.items():
            for packet in stream.encode():
                self.containers[cam].mux(packet)
            self.containers[cam].close()
        self.containers, self.streams = {}, {}


def write_summary(run_dir: Path, args, scene: str, task_desc: str, success_task, t_start: str,
                  episodes: list[dict], videos: dict[str, str]) -> dict:
    """summary.json: aggregated results, videos and per-episode metrics. Rewritten after every
    episode, so an interrupted run also leaves its results."""
    measured = [m for m in episodes if m["success"] is not None]
    successes = [bool(m["success"]) for m in measured]
    results = {
        "n_success": sum(successes),
        "success_rate_pct": pct(rate(measured, "success")),
        "success_at_end_rate_pct": pct(rate(measured, "success_at_end")),
        "per_episode_success": [m["success"] for m in episodes],
        # success rate after each episode (how the rate converges)
        "cumulative_success_rate_pct": [pct(sum(successes[:i + 1]) / (i + 1)) for i in range(len(successes))],
    }
    times = [m["success_time_s"] for m in measured if m.get("success_time_s") is not None]
    results["avg_success_time_s"] = round(float(np.mean(times)), 2) if times else None
    for key in ("side", "blue_side"):
        sides = sorted({m[key] for m in measured if m.get(key)})
        if sides:
            results[f"success_rate_by_{key}"] = {
                sd: {"success_rate_pct": pct(rate(eps, "success")), "episodes": len(eps)}
                for sd in sides for eps in [[m for m in measured if m.get(key) == sd]]}
    summary = {
        "scene": scene, "task": task_desc, "success_task": success_task,
        "success_criterion": task_success.DESCRIPTION.get(success_task) if success_task else None,
        "started": t_start, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_episodes_planned": args.num_episodes, "n_episodes_completed": len(episodes),
        "success_rate": rate(measured, "success"),
        "success_at_end_rate": rate(measured, "success_at_end"),
        "results": results,
        "videos": videos,
        "episodes": episodes,
        "args": {**vars(args), **({"hf_token": "***"} if getattr(args, "hf_token", None) else {})},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# ── Backends ──────────────────────────────────────────────────────────────────
def make_mujoco_robot(args, model_path, scene_objects, scene_joints, control_arms, cameras):
    from lerobot_robot_icub.config_icub_mujoco import iCubMuJoCoConfig
    from lerobot_robot_icub.icub_mujoco import iCubMuJoCo

    robot_cfg = iCubMuJoCoConfig(
        model_path=str(model_path),
        config_path=str(args.config),
        control_arms=control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
        control_dt=1.0 / args.fps,
        vr_enabled=False,
        scene_objects=scene_objects,
        scene_joints=scene_joints,
    )
    if cameras:
        robot_cfg.camera_names = list(cameras)
    return iCubMuJoCo(robot_cfg)


def make_yarp_robot(args, control_arms, cameras):
    from lerobot_robot_icubrg.config_icub import iCubConfig
    from lerobot_robot_icubrg.icub import iCub

    robot_name = args.robot_name or ("icub" if args.backend == "real" else "icubSim")
    os.environ["ICUB_ROBOT_NAME"] = robot_name
    robot_cfg = iCubConfig(
        config_path=str(args.config),
        control_arms=control_arms,
        controller_type=args.controller,
        impedance_mode=args.impedance,
    )
    if cameras:
        missing = robot_cfg.select_cameras(list(cameras))
        if missing:
            print(f"[eval][warn] Cameras of the policy not available on this robot (fed as black): {missing}")
    return iCub(robot_cfg)


def reset_mujoco_episode(robot, ep: int, args) -> dict:
    """Home pose, hands open, scene reset (tools/scene_reset.py) and targets on the hands.
    Returns the layout of the reset (side, initial object poses, door panel center)."""
    import mujoco

    tc = robot.teleop_core
    mode = args.side or args.door_side or args.mirror
    side = (mode if mode in ("right", "left") else ("right" if ep % 2 == 0 else "left")
            if mode == "alternate" else ("right" if np.random.uniform() < 0.5 else "left"))
    tc.data.qpos[:] = tc.home_qpos
    tc.data.ctrl[:] = tc.home_ctrl
    tc._apply_home_pose_to_arrays(tc.data.qpos, tc.data.ctrl)
    tc.data.qvel[:] = 0
    tc._set_hand_level("right", 0.0)
    tc._set_hand_level("left", 0.0)
    mujoco.mj_forward(tc.model, tc.data)
    tc._reset_scenario(side=side, door_xy=args.door_xy)
    tc._snap_targets_to_ee()
    tc.state = "teleop_active"
    # Settle for 0.5 s holding the home pose (fingers reach the open hand, velocities ~0), as in
    # the recordings, where the robot waits at home before an episode starts; otherwise the first
    # observation (fingers at the XML pose, large velocities) is far out of the training data.
    for _ in range(int(0.5 / tc.control_dt)):
        tc._solve_whole_body()
        tc._interpolate_hands()
        tc._step_physics()
    tc._snap_targets_to_ee()
    layout = tc.scene_layout
    # Side only for two-sided scenes (sides / mirror_y in scenes.yaml)
    return {**({"side": layout["side"]} if tc._randomizer.sides_mode else {}),
            "initial_objects": {n: np.round(p, 3).tolist() for n, (p, _) in layout["objects"].items()},
            **({"door_panel_xy": np.round(layout["door_xy"], 3).tolist()} if layout["door_xy"] else {})}


def reset_yarp_episode(robot, args) -> float:
    """Home pose, hands open, objects respawned (Gazebo) or placed by the operator (real).
    Returns the time from which object poses are valid."""
    robot.go_home()
    robot.send_rpc_command("open_right")
    robot.send_rpc_command("open_left")
    if args.backend == "gazebo":
        robot.send_rpc_command("reset_scenario")
        time.sleep(GAZEBO_RESET_WAIT_S)
    else:
        input("[eval] Place the objects for the next episode and press Enter...")
    since = time.time()
    robot.start_teleop()
    time.sleep(0.5)
    robot.reset_command_state()
    return since


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = get_safe_torch_device(args.device)
    if args.impedance and args.controller != "id":
        print("[eval] --impedance only applies to --controller id: ignored.")
        args.impedance = False
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or True

    ckpt_source, name = resolve_checkpoint_source(args)

    # Strictly require control_config.json in checkpoint or dataset_meta (Option 4.B)
    control_config_file = Path(ckpt_source) / "control_config.json"
    if not control_config_file.is_file() and args.dataset_meta:
        dm_path = Path(args.dataset_meta).expanduser().resolve()
        if (dm_path / "meta" / "control_config.json").is_file():
            control_config_file = dm_path / "meta" / "control_config.json"

    if not control_config_file.is_file():
        print(
            f"\n[eval][FATAL ERROR] Checkpoint at '{ckpt_source}' is missing required 'control_config.json'.\n"
            "Uncharacterized models without control metadata are strictly not allowed.\n",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    try:
        with open(control_config_file, "r", encoding="utf-8") as f:
            control_meta = json.load(f)
    except Exception as e:
        print(f"[eval][FATAL ERROR] Failed to parse {control_config_file}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    ctrl_type = control_meta.get("control", {}).get("type", "tsid")
    args.controller = "id" if ctrl_type in ("id", "tsid") else "ik"
    args.impedance = bool(control_meta.get("control", {}).get("impedance_enabled", False))
    print(
        f"[eval] Auto-configured from control_config.json: "
        f"Controller={args.controller.upper()} ({ctrl_type}), Mode={control_meta.get('control', {}).get('mode')}, "
        f"Impedance={args.impedance}"
    )

    if args.impedance and args.controller != "id":
        args.impedance = False

    run_dir = Path(args.out) if args.out else (
        repo_root / "outputs" / "eval" / f"{name}_{args.backend}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "eval.log", "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    safe_argv = [
        "***" if i > 0 and sys.argv[i - 1] == "--hf-token" else arg
        for i, arg in enumerate(sys.argv)
    ]
    print(f"[eval] command: {' '.join(safe_argv)}")

    # ── Policy and its normalization (saved with the checkpoint) ──────────────
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt_source)
    policy_cfg.pretrained_path = ckpt_source
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy_cfg.chunk_size:
            raise SystemExit(f"--n-action-steps must be in 1..{policy_cfg.chunk_size} (chunk_size)")
        policy_cfg.n_action_steps = args.n_action_steps  # policy.reset() rebuilds the action queue
    policy = get_policy_class(policy_cfg.type).from_pretrained(ckpt_source, config=policy_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=ckpt_source)

    state_names, action_names = training_feature_names(
        ckpt_source, dataset_meta=args.dataset_meta, token=hf_token
    )
    if state_names is None:
        print("[eval][warn] Training dataset meta/info.json not found: state/action order falls back "
              "to the robot's feature order. Pass --dataset-meta <dataset_dir_or_hf_repo_id> if the "
              "policy was trained on a curated state subset.")
    cameras = [k.removeprefix("observation.images.") for k in policy_cfg.input_features
               if k.startswith("observation.images.")]
    control_arms = args.control_arms or arms_of(action_names) or infer_arms_from_dim(policy_cfg)

    model_path, default_task, scene_objects, scene_joints, _ = _resolve_scene(args.scene, controller=args.controller)
    task_desc = args.task or default_task
    success_task = task_success.TASK_BY_SCENE.get(model_path.name)

    if args.backend == "mujoco":
        robot = make_mujoco_robot(args, model_path, scene_objects, scene_joints, control_arms, cameras)
    else:
        robot = make_yarp_robot(args, control_arms, cameras)

    print("\n=======================================================")
    print(" Starting Policy Evaluation")
    print(f" Backend:    {args.backend.upper()}")
    print(f" Controller: {args.controller.upper()} {'(Impedance)' if args.impedance else ''}")
    print(f" Scene:      {model_path.name}  |  Task: {task_desc}")
    print(f" Arms:       {control_arms}  |  Cameras: {cameras}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device:     {device}  |  Results: {run_dir}")
    print(f" Actions:    {policy.config.n_action_steps} of each chunk of {policy_cfg.chunk_size}")
    if success_task:
        when = "" if success_task in task_success.SUCCESS_AT_END_ONLY else ", at some step"
        print(f" Success:    {success_task}: {task_success.DESCRIPTION[success_task]}{when}")
    print("=======================================================\n")

    robot.connect()
    obs_keys = set(robot.observation_features)
    robot_action_keys = list(robot.action_features)
    if state_names:
        missing = [n for n in state_names if n not in obs_keys]
        if missing:
            print(f"[eval][warn] The robot does not observe {len(missing)} trained state features "
                  f"{missing[:6]}: fed as 0. Check --scene / --control-arms.")
    if action_names and set(action_names) != set(robot_action_keys):
        print(f"[eval][warn] Action mismatch: trained={action_names}, robot={robot_action_keys}")

    mj_viewer = scene_state = None
    if args.backend == "mujoco":
        tc = robot.teleop_core
        if success_task:
            scene_state = MujocoSceneState(tc.model, success_task)
        if not args.no_viewer:
            import mujoco
            import mujoco.viewer
            # Without the side panels, on the scene's frontview camera
            mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data, show_left_ui=False, show_right_ui=False)
            frontview = mujoco.mj_name2id(tc.model, mujoco.mjtObj.mjOBJ_CAMERA, "frontview")
            if frontview >= 0:
                with mj_viewer.lock():
                    mj_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    mj_viewer.cam.fixedcamid = frontview
            else:
                mj_viewer.cam.azimuth, mj_viewer.cam.elevation, mj_viewer.cam.distance = 135, -20, 2.0
    elif args.backend == "gazebo" and success_task:
        scene_state = GazeboSceneState(success_task)
        late_topics = scene_state.start()
        if late_topics:
            print(f"[eval][warn] Contact topics not advertised yet: {late_topics} (fine for objects "
                  "created on reset; otherwise grasps / contacts read as none).")

    episodes: list[dict] = []
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    videos = None if args.no_video else CameraVideos(run_dir, cameras, args.fps)
    video_files = videos.files() if videos is not None else {}
    if videos is not None:
        print(f"[eval] Saving videos of {cameras} to {run_dir / 'videos'}")
    try:
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            since = 0.0
            reset_info: dict = {}
            if args.backend == "mujoco":
                reset_info = reset_mujoco_episode(robot, ep, args)
            else:
                since = reset_yarp_episode(robot, args)
                if isinstance(scene_state, GazeboSceneState):
                    late = scene_state.wait(since)
                    if late:
                        print(f"[eval][warn] No new pose from Gazebo for {late}: success not measured this episode.")
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            ep_success = task_success.EpisodeSuccess(success_task) if scene_state is not None else None
            side = None
            infer_ms: list[float] = []
            dt = 1.0 / args.fps
            t_episode = time.perf_counter()
            video_start_s = videos.time_s if videos is not None else None

            for step in range(args.episode_len):
                t0 = time.perf_counter()
                obs = robot.get_observation()
                if videos is not None:
                    videos.add(obs)

                if ep_success is not None:
                    s = scene_state(robot.teleop_core.data) if args.backend == "mujoco" else scene_state(since)
                    if s is not None:
                        ep_success.update(step, s)
                        if side is None and success_task == "stack":
                            side = blue_side(s, args.backend)

                names = state_names or [k for k in robot.observation_features if k not in cameras]
                batch = {"observation.state": torch.tensor(
                    [to_float(obs[n]) if n in obs else 0.0 for n in names], dtype=torch.float32).unsqueeze(0)}
                for cam in cameras:
                    img = obs.get(cam)
                    if img is None:
                        continue
                    batch[f"observation.images.{cam}"] = (
                        torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0))
                batch = {k: v.to(device) for k, v in batch.items()}
                batch["task"] = task_desc
                batch = preprocessor(batch)

                t_inf = time.perf_counter()
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(batch))
                infer_ms.append((time.perf_counter() - t_inf) * 1000.0)
                values = action.detach().cpu().numpy().ravel()
                order = action_names if action_names and len(action_names) == len(values) else robot_action_keys
                named = dict(zip(order, values.tolist()))
                robot.send_action({k: float(named.get(k, 0.0)) for k in robot_action_keys})

                if mj_viewer is not None:
                    mj_viewer.sync()
                elif not args.no_viewer and cameras:
                    img = obs.get(cameras[0])
                    if img is not None:
                        cv2.imshow("iCub Policy Eval", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == 27:
                            print("[Eval] ESC pressed -> Stopping.")
                            return
                if step % (5 * args.fps) == 0:
                    print(f"  step {step:4d}/{args.episode_len}")
                time.sleep(max(dt - (time.perf_counter() - t0), 0.0))

            if ep_success is not None:
                # Criterion on the final state too (the last update came before the last action)
                s = scene_state(robot.teleop_core.data) if args.backend == "mujoco" else scene_state(since)
                if s is not None:
                    ep_success.update(args.episode_len, s)
                metrics = ep_success.summary()
            elif args.backend == "real":
                ans = input("[eval] Success? [y/N]: ").strip().lower() == "y"
                metrics = {"success": ans, "success_step": None, "success_at_end": ans}
            else:
                metrics = {"success": None, "success_step": None, "success_at_end": None}
            if side is not None:
                metrics["blue_side"] = side
            metrics = {"episode": ep + 1, **metrics, **reset_info,
                       "success_time_s": (round(metrics["success_step"] / args.fps, 2)
                                          if metrics.get("success_step") is not None else None),
                       "wall_time_s": round(time.perf_counter() - t_episode, 1),
                       # stretch of the episode in videos/<camera>.mp4
                       **({"video_start_s": video_start_s, "video_end_s": videos.time_s}
                          if videos is not None else {}),
                       "infer_ms_mean": float(np.mean(infer_ms)), "infer_ms_max": float(np.max(infer_ms))}
            episodes.append(metrics)
            write_summary(run_dir, args, model_path.name, task_desc, success_task, t_start, episodes, video_files)
            finals = " ".join(f"{k.removeprefix('final_')}={v}" for k, v in metrics.items() if k.startswith("final_"))
            side_str = f" (side {reset_info['side']})" if reset_info.get("side") else f" (blue {side})" if side else ""
            print(f"Episode {ep + 1}{side_str}: "
                  f"success={metrics['success']} (step {metrics['success_step']}), "
                  f"at end={metrics['success_at_end']} {finals}")

        measured = [m for m in episodes if m["success"] is not None]
        summary = write_summary(run_dir, args, model_path.name, task_desc, success_task, t_start,
                                episodes, video_files)
        if measured:
            print(f"\n[eval] success_rate: {summary['success_rate']:.1%}  "
                  f"success_at_end: {summary['success_at_end_rate']:.1%}  ({len(measured)} episodes)")
            by_side = (summary["results"].get("success_rate_by_side")
                       or summary["results"].get("success_rate_by_blue_side", {}))
            for sd, r in by_side.items():
                print(f"[eval]   {sd:5s} side: {r['success_rate_pct']}%  ({r['episodes']} episodes)")
        print(f"[eval] Summary saved to {run_dir / 'summary.json'}")

    finally:
        if videos is not None:  # closes the .mp4 files (also when interrupted) so they open
            videos.close()
        if isinstance(scene_state, GazeboSceneState):
            scene_state.stop()
        if mj_viewer is not None:
            mj_viewer.close()
        robot.disconnect()
        if not args.no_viewer:
            cv2.destroyAllWindows()
        print("[Eval] Completed.")


if __name__ == "__main__":
    main()
