# User manual · Experimentation Hub

**English** | [Español](MANUAL.es.md) · [Back to README](../README.md)

This manual walks through a complete session, from choosing the platform to publishing the
dataset on Hugging Face. Installation and the VR headset setup are covered in the
[README](../README.md#installation-and-startup).

- [Before you start](#before-you-start)
- [Welcome screen](#welcome-screen)
- [1. Configuration](#1-configuration)
- [2. Episode Control](#2-episode-control)
- [3. Visualize & Curate](#3-visualize--curate)
- [Where the data is saved](#where-the-data-is-saved)
- [Troubleshooting](#troubleshooting)

---

## Before you start

- The Hub is running and open at <http://localhost:7860>.
- The Meta Quest 2 has the BeaVR app installed and is connected over Wi-Fi or USB cable
  (see [VR headset](../README.md#vr-headset)).
- For the **real iCub**, the computer is on the robot's network and its YARP name server
  is running.

---

## Welcome screen

Choose the platform you are going to record with. **iCub Simulated MJ** runs the MuJoCo
simulation, and **iCub Real / Gazebo** works with both the Gazebo simulation and the
physical robot.

![Welcome screen](images/inicio.png)

---

## 1. Configuration

### Session settings

Fill in the session parameters.

| Field | What it is |
|---|---|
| **Task** | Scene to record (Block Lifting, Pick-and-Place, …) |
| **Repo ID** | Dataset name, for example `local/icub_block_lifting` |
| **Number of episodes** | How many episodes the session will record |
| **FPS** | Recording frequency (30 by default) |
| **Max duration** | Maximum length of each episode in seconds (0 = no limit) |
| **Data folder** | Where the dataset is saved |
| **Control arms** | Which arms are teleoperated and recorded |
| **VR connection** | USB cable, or the headset's IP for Wi-Fi |

![Configuration in MuJoCo](images/conf_mj_final.png)

### Robot stack (Gazebo / real only)

On the *iCub Real / Gazebo* platform, start the robot side in order before launching the
session.

1. **Start the simulation** (Gazebo mode) or **connect to the robot** (real mode). The
   status indicator turns green when it is ready.
2. **Start the YARP module**, which is the component that commands the robot.

![Configuration in Gazebo / real](images/conf_rg_final.png)

### Launch the session

Press **LAUNCH SESSION** (MuJoCo) or **LAUNCH TELEOP / RECORDING** (Gazebo / real). The
launch status shows whether the session started correctly, and the Hub switches to
*Episode Control*.

---

## 2. Episode Control

### Recording with the VR headset

Put the headset on. You will see the robot's eyes and the front view, along with the
recording status.

| Control | Action |
|---|---|
| **A** | Start recording an episode |
| **B** | Stop recording and save the episode |
| **X** | Pause or resume teleoperation, to reposition yourself without moving the robot |
| **Y** | Reset the scene (repositions the objects) |
| **Grip** (lower trigger, middle finger) | Open and close the hand gradually |

A typical episode goes like this.

1. Press **Y** if you want the objects in a new position.
2. Press **A** and perform the task.
3. Press **B** when you finish, and the episode is saved.
4. Use **X** whenever you need to reposition yourself between episodes.

### The Episode Control tab

The **RECORD episode**, **STOP episode** and **End session** buttons do the same as the
headset controls, which is useful when someone else is assisting the operator. The tab
also shows the session status, the progress (saved episodes out of the planned ones), live
metrics (FPS, latency, dataset size), per-episode metrics and the live log.

In the **Logbook** you can write notes about the current episode, for example if the
grasp failed or something unexpected happened. Each note is saved with the episode number.

![Episode Control](images/episode_control_final.png)

When you have recorded all the episodes, press **End session**.

---

## 3. Visualize & Curate

### Choose the dataset

Select a dataset from the list and load it. From here you can also delete a dataset, which
sends it to the system trash after asking for confirmation.

![Dataset selection](images/dataset_final.png)

### Review the episodes

Browse the episodes with the three cameras playing in sync. A single control bar moves all
of them, and a vertical line follows the video time on each plot. Clicking on a plot jumps
the videos to that instant. Mark the episodes that should be excluded from the final
dataset, for example failed attempts.

![Synchronized player](images/reproductor_sincronizado.png)

### Plots, signals and instruction

- **Signal plots.** Plot any dimension of `action` or `observation.state` over the video,
  and export the plots to HTML or PDF.
- **Signals to include.** Remove whole features or individual dimensions. The search box
  filters the list to find them quickly.
- **Language instruction.** Edit the task instruction for the whole dataset.

![Signals and instruction](images/senales_final.png)

### Package and upload

1. **Package** to generate the curated dataset as a separate copy in `data_curated/`.
2. **Upload to Hugging Face Hub** with your user and token. The *Will upload* line shows
   whether the packaged or the original dataset will be uploaded. If there are curation
   changes that have not been packaged, the upload is blocked until you package them.

![Packaging and upload](images/subir_final.png)

---

## Where the data is saved

| Location | Contents |
|---|---|
| `data/<repo_id>_<date>/` | Recorded dataset in LeRobot format |
| `data/<repo_id>_<date>/hub_session/` | Per-episode metrics and logbook notes |
| `data_curated/` | Curated datasets |
| `hub_dataset/logs/` | Full log of each Hub run |

The upload to Hugging Face includes the `data/`, `meta/` and `videos/` folders.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `adb devices` shows the headset as `unauthorized` | Put the headset on and accept *Allow USB debugging* |
| The headset does not show up in `adb devices` | Check the cable and the udev rule from the [README](../README.md#usb-cable-setup-one-time-only) |
| A VR port is busy when launching | The Hub frees the ports left by its own sessions. If another program holds them, the launch status says which one |
| The Gazebo or MuJoCo window does not open in Docker | Run `xhost +SI:localuser:$(id -un)` before `docker compose up` |
| The robot does not move in Gazebo / real | Check that the simulation (or the robot connection) and the YARP module are both green in *Robot stack* |
| The upload is blocked | There are curation changes without packaging. Package the dataset first |
