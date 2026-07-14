#!/usr/bin/env python3
"""
MuJoCo Dataset Recorder for LeRobot
=====================================

Records datasets for Imitation Learning using the LeRobot library.
All data comes directly from MuJoCo — no YARP ports.

Captures:
  1. RGB Images     rendered from a MuJoCo camera (e.g. front_cam)
  2. Robot State    joint positions for all actuators (38-dim)
    3. World State    relevant object pose(s) in world frame (currently cube pose)
    4. Human Actions  mocap target positions/orientations + gaze target

This module is imported by teleop_mujoco.py.  It can also be used
standalone for offline replay-based recording.

Usage (as library):
    recorder = MuJoCoRecorder(model, data, fps=30, repo_id="demo")
    recorder.start_episode()
    # ...in loop...
    recorder.add_frame(state_vec, action_vec)
    # ...end of episode...
    recorder.stop_episode()
"""

import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ---- Optional heavy imports (graceful fallback) ----
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


class MuJoCoRecorder:
    """
    Records datasets from a running MuJoCo simulation.

    Images are rendered with ``mujoco.Renderer`` using a named camera
    defined in the XML scene.  State & action vectors are passed in
    explicitly by the caller (typically ``teleop_mujoco.py``).
    """

    def __init__(self, model, data, fps=30, repo_id="icub_mujoco",
                 root_dir="data", camera_name="head_cam",
                 img_width=640, img_height=480):
        """
        Args:
            model:       MuJoCo model  (mujoco.MjModel)
            data:        MuJoCo data   (mujoco.MjData)
            fps:         Target frames-per-second for the dataset
            repo_id:     LeRobot dataset name / repository ID
            root_dir:    Base directory for dataset storage
            camera_name: Name of the MuJoCo camera used for observations
            img_width:   Render width (pixels)
            img_height:  Render height (pixels)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MuJoCoRecorder")
        if not LEROBOT_AVAILABLE:
            raise ImportError("lerobot is required for MuJoCoRecorder")
        if not MUJOCO_AVAILABLE:
            raise ImportError("mujoco is required for MuJoCoRecorder")

        self.model = model
        self.data = data
        self.fps = fps
        self.repo_id = repo_id
        self.root_dir = root_dir
        self.camera_name = camera_name
        self.img_w = img_width
        self.img_h = img_height
        self.world_feature_names = [
            "cube_x", "cube_y", "cube_z",
            "cube_qw", "cube_qx", "cube_qy", "cube_qz",
        ]

        # MuJoCo offscreen renderer
        self.renderer = mujoco.Renderer(model, height=img_height, width=img_width)

        # Dataset (lazy init on first start_episode)
        self.dataset = None
        self.dataset_path = None

        # Counters
        self.episode_count = 0
        self.episode_frames = 0
        self._dataset_initialized = False

        logger.info("MuJoCoRecorder ready  "
                     f"(camera={camera_name}, {img_width}×{img_height}, "
                     f"fps={fps})")

    # ------------------------------------------------------------------ #
    #                          Public API                                  #
    # ------------------------------------------------------------------ #

    def start_episode(self):
        """Begin a new recording episode."""
        if not self._dataset_initialized:
            self._init_dataset()
        self.episode_frames = 0

    def add_frame(self, state: np.ndarray, action: np.ndarray, world_state: np.ndarray | None = None):
        """
        Record one frame.

        Args:
            state:  Robot joint state vector  (float32, shape [N])
            action: Human action vector       (float32, shape [M])
        """
        if self.dataset is None:
            return

        # Render image from MuJoCo camera
        self.renderer.update_scene(self.data, camera=self.camera_name)
        image_rgb = self.renderer.render()  # (H, W, 3) uint8 RGB
        image_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1)

        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
        action_tensor = torch.from_numpy(np.asarray(action, dtype=np.float32))

        try:
            if world_state is None:
                world_state = np.zeros(len(self.world_feature_names), dtype=np.float32)
            world_state_tensor = torch.from_numpy(np.asarray(world_state, dtype=np.float32))

            self.dataset.add_frame({
                "observation.image.front": image_tensor,
                "observation.state": state_tensor,
                "observation.state_world": world_state_tensor,
                "action": action_tensor,
                "task": "teleoperate icub mujoco",
            })
            self.episode_frames += 1
            if self.episode_frames % 100 == 0:
                print(f"\r  [REC] Frames: {self.episode_frames}", end="",
                      flush=True)
        except Exception as e:
            logger.error(f"Error adding frame: {e}")

    def stop_episode(self):
        """Save the current episode to disk."""
        if self.dataset is None or self.episode_frames == 0:
            logger.warning("stop_episode: nothing to save.")
            return

        try:
            self.dataset.save_episode()
            print(f"\n  [REC] Episode {self.episode_count} saved "
                  f"({self.episode_frames} frames)")
            self.episode_count += 1
        except Exception as e:
            logger.error(f"Error saving episode: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    #                       Dataset initialisation                         #
    # ------------------------------------------------------------------ #

    def _init_dataset(self):
        """Create the LeRobotDataset structure (called once)."""
        # Infer dimensions from the model
        STATE_DIM = self.model.nu       # one entry per actuator
        WORLD_DIM = len(self.world_feature_names)
        ACTION_DIM = 19                 # rh_pos(3)+rh_quat(4)+rh_gripper(1)+lh_pos(3)+lh_quat(4)+lh_gripper(1)+gaze_pos(3)

        features = {
            "observation.image.front": {
                "dtype": "video",
                "shape": (3, self.img_h, self.img_w),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": [
                    mujoco.mj_id2name(self.model,
                                      mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    or f"act_{i}"
                    for i in range(STATE_DIM)
                ],
            },
            "observation.state_world": {
                "dtype": "float32",
                "shape": (WORLD_DIM,),
                "names": self.world_feature_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": [
                    "rh_pos_x", "rh_pos_y", "rh_pos_z",
                    "rh_quat_w", "rh_quat_x", "rh_quat_y", "rh_quat_z",
                    "rh_gripper",
                    "lh_pos_x", "lh_pos_y", "lh_pos_z",
                    "lh_quat_w", "lh_quat_x", "lh_quat_y", "lh_quat_z",
                    "lh_gripper",
                    "gaze_x", "gaze_y", "gaze_z",
                ],
            },
        }

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dataset_path = Path(self.root_dir) / f"{self.repo_id}_{timestamp}"

        try:
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.dataset_path,
                robot_type="icub",
                features=features,
                use_videos=True,
            )
            self._dataset_initialized = True
            logger.info(f"Dataset created at {self.dataset_path}")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            import traceback
            traceback.print_exc()
