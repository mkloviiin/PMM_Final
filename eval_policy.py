#!/usr/bin/env python3
"""Unified policy evaluation for trained LeRobot VLA models across MuJoCo, Gazebo, and Real iCub.

Supported Backends:
  - mujoco: Evaluates policy inside MuJoCo physics with TSIK (ik) or TSID (id).
  - gazebo: Evaluates policy against Gazebo Sim via YARP / teleop_module_sm.
  - real: Evaluates policy against the Real iCub robot via YARP / teleop_module_sm.

Success: robosuite's _check_success of the task (metrics/task_success.py), the same code for
every simulator: success if it holds at some step (as robomimic rollouts) and also at the end
of the episode. MuJoCo builds the scene state from MjData; Gazebo from its topics (object poses,
fingertip contact sensors, door joint states); on the real robot the operator answers.

Each episode starts at home with the hands open and the scene reset (scenes.yaml ranges).
Results: outputs/eval/<checkpoint>_<backend>_<date>/ (eval.log, summary.json).

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
        help="Path or HuggingFace repo ID for the trained policy checkpoint",
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
                        help="Door scene (MuJoCo): handle side (default: XML, left); alternate = "
                             "right in odd episodes, left in even ones")
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


def training_feature_names(checkpoint: Path) -> tuple[list[str] | None, list[str] | None]:
    """State / action names of the training dataset (train_config.json -> dataset meta/info.json)."""
    try:
        cfg = json.loads((checkpoint / "train_config.json").read_text())
        feats = json.loads((Path(cfg["dataset"]["root"]).expanduser() / "meta" / "info.json").read_text())["features"]
        return feats["observation.state"]["names"], feats["action"]["names"]
    except Exception:  # noqa: BLE001
        return None, None


def arms_of(action_names: list[str] | None) -> str | None:
    if not action_names:
        return None
    has_r = any(n.startswith("rh_") for n in action_names)
    has_l = any(n.startswith("lh_") for n in action_names)
    return "both" if has_r and has_l else "right" if has_r else "left" if has_l else None


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


def reset_mujoco_episode(robot, ep: int, args) -> None:
    """Home pose, hands open, scene reset (scenes.yaml ranges) and targets on the hands."""
    import mujoco

    tc = robot.teleop_core
    if args.door_side is not None:
        from tools.door_scene import set_handle_side
        set_handle_side(tc.model, args.door_side if args.door_side != "alternate"
                        else ("right" if ep % 2 == 0 else "left"))
    tc.data.qpos[:] = tc.home_qpos
    tc.data.ctrl[:] = tc.home_ctrl
    tc._apply_home_pose_to_arrays(tc.data.qpos, tc.data.ctrl)
    tc.data.qvel[:] = 0
    tc._set_hand_level("right", 0.0)
    tc._set_hand_level("left", 0.0)
    mujoco.mj_forward(tc.model, tc.data)
    tc._reset_scenario()
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

    ckpt = Path(args.checkpoint).expanduser()
    name = ckpt.parent.parent.name if ckpt.is_dir() and ckpt.name == "pretrained_model" else ckpt.name
    run_dir = Path(args.out) if args.out else (
        repo_root / "outputs" / "eval" / f"{name}_{args.backend}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "eval.log", "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    print(f"[eval] command: {' '.join(sys.argv)}")

    # ── Policy and its normalization (saved with the checkpoint) ──────────────
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_cfg.pretrained_path = str(ckpt)
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy_cfg.chunk_size:
            raise SystemExit(f"--n-action-steps must be in 1..{policy_cfg.chunk_size} (chunk_size)")
        policy_cfg.n_action_steps = args.n_action_steps  # policy.reset() rebuilds the action queue
    policy = get_policy_class(policy_cfg.type).from_pretrained(str(ckpt), config=policy_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(ckpt))

    state_names, action_names = training_feature_names(ckpt)
    if state_names is None:
        print("[eval][warn] Training dataset meta/info.json not found: state/action order falls back "
              "to the robot's feature order.")
    cameras = [k.removeprefix("observation.images.") for k in policy_cfg.input_features
               if k.startswith("observation.images.")]
    control_arms = args.control_arms or arms_of(action_names) or "right"

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
        print(f" Success:    {success_task}: {task_success.DESCRIPTION[success_task]}, at some step")
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
        if args.door_xy is not None:
            from tools.door_scene import place_door
            import mujoco
            place_door(tc.model, tc.data, args.door_xy)
            mujoco.mj_forward(tc.model, tc.data)
        if success_task:
            scene_state = MujocoSceneState(tc.model, success_task)
        if not args.no_viewer:
            import mujoco.viewer
            mj_viewer = mujoco.viewer.launch_passive(tc.model, tc.data)
            mj_viewer.cam.azimuth, mj_viewer.cam.elevation, mj_viewer.cam.distance = 135, -20, 2.0
    elif args.backend == "gazebo" and success_task:
        scene_state = GazeboSceneState(success_task)
        late_topics = scene_state.start()
        if late_topics:
            print(f"[eval][warn] Contact topics not advertised yet: {late_topics} (fine for objects "
                  "created on reset; otherwise grasps / contacts read as none).")

    episodes: list[dict] = []
    try:
        for ep in range(args.num_episodes):
            print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
            since = 0.0
            if args.backend == "mujoco":
                reset_mujoco_episode(robot, ep, args)
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
            infer_ms: list[float] = []
            dt = 1.0 / args.fps

            for step in range(args.episode_len):
                t0 = time.perf_counter()
                obs = robot.get_observation()

                if ep_success is not None:
                    s = scene_state(robot.teleop_core.data) if args.backend == "mujoco" else scene_state(since)
                    if s is not None:
                        ep_success.update(step, s)

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
            metrics.update(episode=ep + 1, infer_ms_mean=float(np.mean(infer_ms)), infer_ms_max=float(np.max(infer_ms)))
            episodes.append(metrics)
            finals = " ".join(f"{k.removeprefix('final_')}={v}" for k, v in metrics.items() if k.startswith("final_"))
            print(f"Episode {ep + 1}: success={metrics['success']} (step {metrics['success_step']}), "
                  f"at end={metrics['success_at_end']} {finals}")

        measured = [m for m in episodes if m["success"] is not None]
        summary = {
            "args": vars(args), "scene": model_path.name, "task": task_desc,
            "success_task": success_task,
            "success_criterion": task_success.DESCRIPTION.get(success_task) if success_task else None,
            "episodes": episodes,
            "success_rate": sum(m["success"] for m in measured) / len(measured) if measured else None,
            "success_at_end_rate": sum(m["success_at_end"] for m in measured) / len(measured) if measured else None,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        if measured:
            print(f"\n[eval] success_rate: {summary['success_rate']:.1%}  "
                  f"success_at_end: {summary['success_at_end_rate']:.1%}  ({len(measured)} episodes)")
        print(f"[eval] Summary saved to {run_dir / 'summary.json'}")

    finally:
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
