# Experimentation Hub

**English** | [Español](README.es.md)

Web platform to **record, review, curate and publish robotics datasets** in
[LeRobot](https://github.com/huggingface/lerobot) format. Designed for **any robot**:
recording, episode review, curation and the upload to Hugging Face are shared, and each
robot plugs in as an independent *backend*
(see [How to add a robot](#how-to-add-a-robot)).

It currently has one integrated robot, the **iCub**, on three platforms: the **MuJoCo**
simulation, the **Gazebo** simulation and the **real iCub**. All three are teleoperated
with a virtual reality headset (Meta Quest 2), using a modified version of the **BeaVR**
app.

- [Available platforms](#available-platforms)
- [Installation and startup](#installation-and-startup)
- [How to use it](#how-to-use-it)
- [Architecture](#architecture)
- [How to add a robot](#how-to-add-a-robot)
- [Recorded data](#recorded-data)
- [Repository structure](#repository-structure)
- [Related documentation](#related-documentation)

---

## Available platforms

| Platform (welcome screen) | Backend | How it runs | Tasks |
|---|---|---|---|
| **iCub Simulated MJ** | `robots/icub_sim.py` | Inside the Hub process (MuJoCo 3.3) | Block Lifting, Block Stacking, Pick-and-Place, Door Opening, Two-Arm Peg-In-Hole, Two-Arm Lifting |
| **iCub Real / Gazebo** | `robots/icub_rg.py` | Separate subprocess, connected to the robot through YARP | Block Lifting, Block Stacking, Pick-and-Place, Door Opening, Two-Arm Peg-In-Hole, Two-Arm Lifting |

- Tasks are declared once in
  `icub-lerobot-mj/dependencies/assets/scenes/scenes.yaml`, and Gazebo uses the same file.
- Both backends record **the same dataset**: same variables, in the same frame and with
  the same cameras. A dataset from one platform therefore works for the other (see
  [Recorded data](#recorded-data)).

**Teleoperation:** with the BeaVR app on a Meta Quest, over Wi-Fi or a USB cable.

| VR control | Action |
|---|---|
| Lower trigger (grip) | Gripper closure, gradual: 0 = open, 1 = closed |
| Index trigger | Freezes arms and head (to reposition yourself) |
| **A** | Start episode |
| **B** | Stop and save episode (hold **B**: discard it) |
| **Y** | Reposition the scene objects |

Without VR, MuJoCo also accepts the viewer's keyboard: `8`/`9` open and close the right
hand, `6`/`7` the left one, and `N` repositions the objects.

---

## Installation and startup

### With Docker (recommended)

A single image contains the Hub, MuJoCo, Gazebo and the YARP client for the real robot.
It requires Linux with X11, an NVIDIA GPU with the NVIDIA Container Toolkit and about
25 GB of free space.

```bash
docker compose build                  # the first time; afterwards only when the code changes
xhost +SI:localuser:$(id -un)         # lets the container open windows
docker compose up                     # starts the Hub
```

Then open <http://localhost:7860>. Datasets are saved in the host's `./data` and
`./data_curated`. Details (volumes, real robot, VR ports, updating the environment) in
[docker/README.md](docker/README.md).

### Without Docker

The environment is `icubenv`, with Python 3.12, YARP 3.12, gz-sim 10, MuJoCo 3.3,
torch 2.7 and LeRobot v0.5.0+18. It is reproduced exactly from the lockfiles used by the
image:

```bash
conda create -n icubenv --file docker/conda-linux-64.lock
conda activate icubenv
pip install --no-deps --ignore-installed -r docker/requirements.lock.txt
pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@017ff73fbfe46bf9a673cd9b402988dcb79151f7"
python hub_dataset/hub_experimentacion.py                       # http://localhost:7860
```

The patched Gazebo controlboard (`icub-lerobot-gazebo/gazebo/plugins/*.so`) comes already
compiled for this environment. If it does not load, rebuild it with
`bash icub-lerobot-gazebo/gazebo/plugins/build_controlboard.sh`.

---

## How to use it

The Hub follows the order of its tabs, which are numbered.

**Welcome screen.** Choose the platform: *iCub Simulated MJ* or *iCub Real / Gazebo*.

**1. Configuration**

- *1. Session settings*. Set:
  - the task and the `Repo ID` (dataset name);
  - the number of episodes, the FPS and the maximum duration of each episode (0 = no limit);
  - the folder where the dataset is saved and the arms to control;
  - the VR: over USB cable or through the Quest's IP.
- *Gazebo / real only*: *2. Robot stack*. Start, in order:
  1. the Gazebo simulation (or the connection to the real robot);
  2. the YARP module.
- **Launch the session**: *LAUNCH SESSION* in MuJoCo or *LAUNCH TELEOP / RECORDING* in Gazebo/real.

**2. Episode Control**

- Session status and progress (saved episodes out of the planned ones).
- *RECORD episode*, *STOP episode* and *End session* buttons. They do the same as A and B on the VR.
- *Logbook*: notes about the current episode, saved with its number.
- Live metrics (FPS, latency, dataset size) and per-episode metrics.
- Live log. The full log is kept in `hub_dataset/logs/`.

**3. Visualize & Curate**

1. *Dataset*: choose, load or delete a dataset. Deleting sends it to the system trash.
2. *Episodes*: browse the episodes with their three synchronized cameras and mark the ones
   excluded from the final dataset.
3. *Signal plots*: plot any dimension of `action` or `observation.state` over the video.
   Plots can be exported to HTML or PDF.
4. *Signals to include*: remove whole features or individual dimensions.
5. *Language instruction*: edit the task instruction for the whole dataset.
6. *Packaging*: generates the curated dataset in `data_curated/`. The original is not modified.
7. *Upload to Hugging Face Hub*: uploads the packaged dataset if there is one; otherwise,
   the original as recorded.
   - The *Will upload* line shows which of the two will be uploaded.
   - If there are curation changes that are not packaged, the upload is blocked.
   - Only `data/`, `meta/` and `videos/` are uploaded.

Each dataset is saved in `data/<repo_id>_<date>/` and includes a `hub_session/` subfolder with:

- `episode_metrics.json/.csv/.png`: FPS, latency, duration and size of each episode;
- `logbook.json/.txt`: the logbook notes.

---

## Architecture

```mermaid
flowchart LR
    U[Browser<br/>localhost:7860] --> APP

    subgraph HUB["hub_dataset/launcher_gui (Gradio)"]
        APP[app.py<br/>welcome + tabs] --> REG[robots/registry.py]
        APP --> SES[session.py<br/>recording session]
        APP --> CUR[curation.py<br/>visualize and curate]
        APP --> PUSH[hub_push.py<br/>upload to Hugging Face]
        REG --> SIM[robots/icub_sim.py]
        REG --> RG[robots/icub_rg.py]
    end

    SIM -- "record_fn (same process)" --> PMJ[icub-lerobot-mj/play_mujoco.py<br/>MuJoCo + iKin IK]
    RG -- "record_fn → subprocess" --> PGZ[icub-lerobot-gazebo/play_gazebo.py]
    PGZ <-->|YARP| YM[teleop_module_sm.py<br/>YARP module]
    YM <--> ROBOT[Gazebo / real iCub]
    VR[Meta Quest + BeaVR] -. ZMQ .-> PMJ
    VR -. ZMQ .-> PGZ

    PMJ --> DS[(LeRobot dataset<br/>data/)]
    PGZ --> DS
    DS --> CUR --> DC[(Curated dataset<br/>data_curated/)]
    DS --> PUSH
    DC --> PUSH --> HF[(Hugging Face Hub)]
```

### Hub modules (`hub_dataset/launcher_gui/`)

| Module | What it does |
|---|---|
| `app.py` | Builds the interface (welcome screen, tabs, CSS) and wires each component to its logic |
| `robots/base.py`, `robots/registry.py` | Backend contract and list of available backends |
| `robots/icub_sim.py`, `robots/icub_rg.py` | Form and launch of each platform |
| `session.py` | Session thread, command queue, status, log and the *Episode Control* poller |
| `session_log.py` | Per-episode metrics and logbook (the dataset's `hub_session/`) |
| `curation.py` | Dataset loading, episode navigation, plots, deletion and packaging |
| `hub_push.py` | Upload to Hugging Face Hub |
| `video_sync.py` | Player for the three cameras, synchronized with the plots |
| `procman.py` | Long-lived external processes (e.g. the YARP module) and scripts (`start_sim.sh`) |
| `ports.py` | Checks and frees the VR ports before launching a session |
| `state.py` | State shared between modules |
| `ui/` | Construction of each tab, folder picker and numeric field filters |

Only **one recording session can be active at a time**.

---

## How to add a robot

The shared tabs do not need to be touched. A new file in `robots/` and one line in the
registry are enough.

**1. Create `hub_dataset/launcher_gui/robots/my_robot.py`:**

```python
from datetime import datetime
from pathlib import Path

import gradio as gr

from .. import session
from ..paths import ASSETS_DIR, HUB_ROOT
from .base import RobotBackend


def build_config_form() -> dict[str, gr.components.Component]:
    """Components of the Configuration tab (already inside a gr.Column)."""
    gr.Markdown("### 1. Session settings")
    repo_id = gr.Textbox(value="local/my_robot_demo", label="Repo ID")
    num_eps = gr.Number(value=50, label="Number of episodes", precision=0, minimum=1)
    gr.Markdown("### 2. Launch session")
    launch_btn = gr.Button("LAUNCH SESSION", variant="primary", size="lg")
    launch_status = gr.Textbox(label="Launch status", interactive=False)
    # num_eps, launch_btn and launch_status are required: app.py uses them by name
    return dict(repo_id=repo_id, num_eps=num_eps,
                launch_btn=launch_btn, launch_status=launch_status)


def launch(repo_id, num_eps) -> str:
    """Receives the form values in the order of config_input_order."""
    if session.state.running:
        return "A session is already running."
    rid = repo_id.strip() if "/" in repo_id else f"local/{repo_id.strip()}"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_root = HUB_ROOT.parent / "data" / f"{rid.replace('/', '_')}_{stamp}"

    def record_fn(cmd_source, on_status):
        # Connect the robot, create the LeRobotDataset in dataset_root and, in a loop:
        #   cmd = cmd_source()   -> "1" start, "2" save, "3" exit, None
        #   on_status("waiting") / on_status("recording") / on_status(f"saved:{n}")
        #   on_status({"metrics": {"fps": ..., "latency_ms": ..., "size_mb": ...}})
        ...

    return session.start_session(record_fn, repo_id=rid, dataset_root=dataset_root,
                                 num_episodes=int(num_eps))


BACKEND = RobotBackend(
    id="my_robot",
    label="My Robot",                       # button text on the welcome screen
    image=ASSETS_DIR / "my_robot.png",      # button image
    available=True,                         # False: the button only says "not yet available"
    build_config_form=build_config_form,
    launch=launch,
    config_input_order=["repo_id", "num_eps"],  # form keys, in the order of launch()
)
```

**2. Register it** in [robots/registry.py](hub_dataset/launcher_gui/robots/registry.py):

```python
from . import icub_rg, icub_sim, my_robot

REGISTRY = [icub_sim.BACKEND, icub_rg.BACKEND, my_robot.BACKEND]
```

**3. Add its image** to `hub_dataset/assets/` and, if its code lives in a sibling project
(like `icub-lerobot-mj/`), its path in [paths.py](hub_dataset/launcher_gui/paths.py).

With that, the platform shows up on the welcome screen, has its form in *Configuration*,
and *Episode Control*, *Visualize & Curate* and the upload work unchanged.

**Recommendations:**

- **Choose the execution pattern.**
  - If the robot's dependencies can coexist with the Hub's, `record_fn` can run in the
    same process (like `icub_sim`).
  - Otherwise, use a subprocess with an input/output bridge, like `_bridge_play` in
    `icub_rg.py`.
- **Long-lived auxiliary processes** (drivers, servers): launch them with `procman.py`, so
  their output shows up in the Hub log.
- **Datasets interchangeable with the iCub ones:** use the same variable names, frames and
  cameras (see [Recorded data](#recorded-data)).
- **Docker:** if the robot needs new packages, install them in `icubenv`, regenerate the
  lockfiles with `docker/freeze_env.sh` and rebuild the image.

---

## Recorded data

Both iCub backends record the same variables, so a MuJoCo dataset and a Gazebo dataset
are interchangeable:

- **Action.** Target pose of each hand: position and quaternion, in the iCub *root* frame.
  Plus the gripper closure (from 0 to 1) and the gaze point.
- **State.**
  - joints: position, velocity and torque;
  - end-effector pose and last command sent (target pose, gripper and gaze);
  - touch on the thumb, index and middle fingers;
  - pose of the scene objects: simulator *ground truth*, not available on the real robot.
- **Cameras.** `left` and `right` are the iCub's eyes; `front` is an external camera.
  All of them record at 320×240.
- **Only the arm chosen** in *Control arms* is recorded.
- **Quaternions with a fixed sign.** Hand with `quat_y ≤ 0` and objects with `quat_w ≥ 0`,
  in both simulators. This avoids jumps between `q` and `−q`.
- **Difference between platforms.** Gazebo and the real robot also record the eyes
  (`eyes_*`), which MuJoCo does not model.

The full contract, including the differences that remain between simulators, is in the
*Sim-to-sim* section of [icub-lerobot-gazebo/README.md](icub-lerobot-gazebo/README.md)
(in Spanish).

---

## Repository structure

```
PMM_Final/
├── hub_dataset/                 # the Hub
│   ├── hub_experimentacion.py   # entry point (python hub_dataset/hub_experimentacion.py)
│   ├── launcher_gui/            # Hub code (see Architecture)
│   ├── assets/                  # logos and images of the welcome screen
│   └── logs/                    # full log of each run
├── icub-lerobot-mj/             # MuJoCo backend: simulation, IK, VR teleoperation, LeRobot plugins
├── icub-lerobot-gazebo/         # Gazebo / real robot backend: Gazebo world, YARP module, LeRobot plugins
├── metrics/                     # latency, jitter and synchronization of the recordings
├── docker/                      # Dockerfile, environment lockfiles and their README
├── docker-compose.yml
├── data/                        # recorded datasets (not versioned)
└── data_curated/                # curated datasets (not versioned)
```

---

## Related documentation

| Document | Contents |
|---|---|
| [docker/README.md](docker/README.md) | Docker image: requirements, volumes, real robot, VR ports, updating the environment |
| [icub-lerobot-gazebo/README.md](icub-lerobot-gazebo/README.md) | Gazebo / real robot stack (YARP), configuration and *sim-to-sim* data contract |
| [icub-lerobot-gazebo/gazebo/README.md](icub-lerobot-gazebo/gazebo/README.md) | Scripts to start and stop the Gazebo simulation |
| [icub-lerobot-mj/lerobot-robot-icub/README.md](icub-lerobot-mj/lerobot-robot-icub/README.md) | LeRobot plugin for the iCub in MuJoCo |
| [metrics/README.md](metrics/README.md) | How to measure latency, jitter and synchronization of the recordings |
