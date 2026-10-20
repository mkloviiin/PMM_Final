# Docker — Hub de experimentación iCub

Una sola imagen con el Hub (Gradio), MuJoCo, Gazebo y el cliente YARP del robot
real. El entorno es una copia exacta de `icubenv` (Python 3.12, YARP 3.12.1,
gz-sim 10, MuJoCo 3.3.2, torch 2.7.1+cu126, lerobot v0.5.0+18 en el commit
`017ff73f`).

## Requisitos del host

- Linux con X11 (probado en Ubuntu 22.04).
- GPU NVIDIA con driver instalado y **NVIDIA Container Toolkit**
  (`docker info` debe mostrar el runtime `nvidia`).
- Docker con Compose v2 (`docker compose version`).
- ~25 GB libres para la imagen.

## Uso

Desde la raíz del repo:

```bash
docker compose build                 # la primera vez (descarga ~15 GB)
xhost +SI:localuser:$(id -un)         # permite al contenedor abrir ventanas
docker compose up                    # levanta el Hub
```

Abrir <http://localhost:7860>. Para detenerlo: `Ctrl+C` o `docker compose down`.

Una terminal dentro del contenedor (con `icubenv` activo), por ejemplo para ver
la sesión tmux de Gazebo:

```bash
docker compose exec hub bash
tmux attach -t icub_sim
```

## Qué se monta del host

| Host | Contenedor | Para qué |
|---|---|---|
| `./data`, `./data_curated` | `/workspace/PMM_Final/...` | datasets grabados y curados |
| `./outputs`, `./output` | `/workspace/PMM_Final/...` | entrenamientos y checkpoints |
| `./hub_dataset/logs` | `/workspace/PMM_Final/hub_dataset/logs` | logs del Hub y métricas |
| `~/.cache/huggingface` | `/home/icub/.cache/huggingface` | login de HF, tema de Gradio, modelos |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | pantalla (GUI de Gazebo, visor MuJoCo) |

El código va **dentro** de la imagen: si cambias el código, vuelve a correr
`docker compose build` (solo se rehace la última capa, es rápido).

## Robot real

El contenedor usa la red del host (`network_mode: host`), así que alcanza al
robot igual que el PC. El Hub hace `yarp conf <ip> <puerto>` + `yarp detect` al
lanzar en modo *Real Robot*. Por defecto `10.0.0.1:10000`; para cambiarlo:

```bash
ICUB_YARP_IP=10.0.0.2 docker compose up
```

No corras un `yarpserver` en el host al mismo tiempo que Gazebo en el
contenedor: ambos usan el puerto 10000.

## Puertos del VR

MuJoCo y Gazebo/robot real usan los mismos puertos para los lentes: 8115-8121 (entrada del
VR) y 10505/15001 (streaming de imagen). Con `network_mode: host` también chocan con
procesos del host (otro Hub abierto fuera del contenedor, un `teleop_module_sm.py`, etc.).
Antes de lanzar una sesión el Hub revisa los que necesita
(`hub_dataset/launcher_gui/ports.py`): si los retiene una sesión anterior del propio Hub
los libera; si los tiene otro proceso, muestra cuál en vez de lanzar la sesión. No hace
falta reiniciar el Hub para pasar de MuJoCo a Gazebo o al revés.

## LeRobot en la imagen y en el host

La imagen trae lerobot en `017ff73f` (v0.5.x), el mismo que el `icubenv` del host con el que
corre el Hub y se graban los datasets. El entrenamiento (notebook `training_smolvla.ipynb`)
corre en el host con el submódulo `lerobot/` (v0.6.x, entorno `smolvla`). Los scripts de
grabación y de evaluación (`play_*.py`, `eval_*.py`) funcionan con ambas versiones.

## Gazebo

`start_sim.sh` usa el controlboard parchado (modo MIXED para los ojos de
iKinGazeCtrl) que se compila **dentro de la imagen** desde
`icub-lerobot-gazebo/gazebo/plugins/controlboard_mixed_mode.patch`, en
`/opt/gzyarp-patched` (variable `GZ_PATCHED_PLUGIN_DIR`). El `.so` del repo no se
usa.

En un notebook con GPU híbrida (PRIME), quita la línea `GZ_RENDER_ENV: ""` de
`docker-compose.yml` para que Gazebo renderice con la NVIDIA.

## Actualizar el entorno

Los lockfiles salen del `icubenv` del PC de desarrollo. Si instalas o actualizas
paquetes ahí:

```bash
docker/freeze_env.sh               # regenera conda-linux-64.lock y requirements.lock.txt
docker compose build
```

Si cambia la versión de `gz-sim-yarp-plugins`, revisa que el parche siga
aplicando (`build_controlboard.sh` clona la v0.5.3).

## Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `ICUB_YARP_IP` / `ICUB_YARP_PORT` | `10.0.0.1` / `10000` | name server YARP del robot real |
| `GZ_RENDER_ENV` | variables PRIME | prefijo de `gz sim` en `start_sim.sh` |
| `GZ_PATCHED_PLUGIN_DIR` | `gazebo/plugins` del repo | carpeta del controlboard parchado |
| `ICUBENV_SITE_PACKAGES` | `~/miniconda3/envs/icubenv/...` | site-packages extra de los scripts |
| `ICUBENV_PREFIX` | `~/miniconda3/envs/icubenv` | entorno del `gz` que usan `reset_objects.sh` y los lectores de pose del cubo y tacto (`gz_pose.py`) |
| `CONDA_ENV` | `icubenv` | entorno que activan `start_sim.sh` y `build_controlboard.sh` |
