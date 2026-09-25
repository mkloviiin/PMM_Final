# iCub + LeRobot — Gazebo / Robot Real

Integración estilo plugin de LeRobot para iCub, para teleoperación en VR y
grabación de datasets **contra el robot real o contra Gazebo** (no MuJoCo
como física — MuJoCo se usa únicamente como visualizador/solver de IK del
lado del teleoperador). Es la contraparte de
[`icub-lerobot-mj`](../icub-lerobot-mj) (que sí simula la física del robot
en MuJoCo); ambas comparten el mismo teleoperador VR.

## Arquitectura

```
┌────────────────┐                                ┌──────────────────────────┐
│ play_gazebo.py │───────────────────────────────►│ iCubTeleop (MuJoCo view) │
└───────┬────────┘          action targets        └──────────────┬───────────┘
        │                                                        │ feedback visual
        │ observation/action (YARP)                              │
        ▼                                                        │
┌──────────────────────────┐     YARP ports / RPC     ┌─────────▼────────────────┐
│ iCub (lerobot_robot_icubrg) │◄────────────────────────►│ common/yarp_module/            │
│  — cliente YARP           │                          │ teleop_module_sm.py      │
└──────────────────────────┘                          └──────────────┬────────────┘
                                                                       │ YARP (control boards,
                                                                       │  cartesian, gaze)
                                                                       ▼
                                                        ┌──────────────────────────┐
                                                        │ iCub real  ó  Gazebo/YARP │
                                                        └──────────────────────────┘
```

`play_gazebo.py` nunca habla directo con los control boards YARP: siempre
pasa por `common/yarp_module/teleop_module_sm.py`, que es quien controla el robot
(real o simulado) de verdad.

## Dos componentes ejecutables del lado del robot

Antes de correr `play_gazebo.py` hace falta tener corriendo, en este orden:

### 1. El robot — real o simulado (`gazebo/`)

- **Robot real**: nada que lanzar aquí — solo asegúrate de que el iCub real
  esté encendido y su `yarprobotinterface` corriendo, y pon
  `robot_name: "icub"` en `config/control_config.yaml`.
- **Robot simulado (Gazebo)**: lanza la simulación completa (yarpserver,
  Gazebo, `yarprobotinterface`, solvers cartesianos y gaze) con:

  ```bash
  cd gazebo/scripts
  chmod +x start_sim.sh stop_sim.sh   # solo la primera vez
  ./start_sim.sh
  ```

  Esto abre una sesión `tmux` (`icub_sim`) con seis ventanas. Usa
  `tmux attach -t icub_sim` para inspeccionarlas y `./stop_sim.sh` para
  cerrarlo todo ordenadamente. Detalle completo en
  [`gazebo/README.md`](gazebo/README.md). Asegúrate de
  que `robot_name: "icubSim"` en `config/control_config.yaml`.

### 2. El YARP RFModule (`common/yarp_module/teleop_module_sm.py`)

Con el robot (real o Gazebo) ya arriba:

```bash
python common/yarp_module/teleop_module_sm.py
```

Este módulo es el mismo para ambos casos — el único cambio entre robot real
y simulado es el campo `robot_name` en `config/control_config.yaml`, que
determina los puertos de cámara (`/icub/cam/...` vs `/icubSim/cam/...`) y
qué instancia YARP controla. El módulo:
- Se conecta a los control boards, al Cartesian Controller y al Gaze
  Controller del robot vía `icubyarpinterface.py`.
- Abre el puerto RPC `/teleop/rpc:i` y los puertos de estado/cámara que
  usa `lerobot_robot_icubrg.iCub` como cliente.

## 3. Grabación / teleoperación LeRobot (`play_gazebo.py`)

```bash
python play_gazebo.py \
  --vr-ip <IP_DE_TUS_QUEST> \
  --repo-id mi_dataset_gazebo
```

**Argumentos** (idénticos a `icub-lerobot-mj/play_mujoco.py`, salvo la pose
del cubo — ver más abajo — y dos flags propios de YARP):

| Argumento | Default | Descripción |
| :--- | :--- | :--- |
| `--repo-id` | `local/icub_gazebo_demo` | Nombre del dataset |
| `--root` | `../data` | Directorio raíz del dataset |
| `--fps` | `30` | FPS del dataset |
| `--num-episodes` | `50` | Número de episodios a grabar |
| `--single-task` | `"Pick up the blue cube"` | Descripción de la tarea |
| `--episode-time-s` | `0` | Duración del episodio en segundos (0 = manual) |
| `--config` | `config/control_config.yaml` | Ruta al config compartido |
| `--model` | `mujoco/assets/scenes/scene_icub_empty_table.xml` | Escena MuJoCo del visualizador |
| `--vr` | `False` | Habilitar control VR |
| `--vr-ip` | `None` | IP del Quest para ZMQ |
| `--vr-cable` | `False` | Conectar el Quest por USB (adb reverse) en vez de WiFi |
| `--push-to-hub` | `False` | Subir dataset a HuggingFace Hub |
| `--control-arms` | `auto` | `auto`, `right`, `left`, `both` — detecta brazos disponibles en YARP |
| `--test-camera` | `False` | Suscribirse a las cámaras del robot |
| `--no-record` | `False` | Solo teleoperar, sin grabar dataset |

Sin `--vr`/`--vr-ip`: controles de consola (`1`=iniciar episodio,
`2`=detener y guardar, `3`=salir). Con VR: botón **A**=iniciar, **B**=detener
y guardar, **B mantenido**=descartar episodio.

> **Nota:** por ahora solo hay un escenario cableado — levantar el cubo
> azul (`scene_icub_lift.xml`, tarea `"Pick up the blue cube"`). No existe
> flag `--scene`; para otra tarea hay que pasar `--model`/`--single-task`
> manualmente, o extender `SCENE_MODEL`/`SCENE_TASK`/`SCENE_OBJECTS` en
> `play_gazebo.py` cuando haya más de un escenario listo para Gazebo.

## Diferencia con `icub-lerobot-mj`: sin pose del cubo

`icub-lerobot-mj` graba `object_pos_x/y/z` y `object_quat_w/x/y/z` porque el
robot **es** la simulación MuJoCo, que conoce la pose exacta del cubo en
todo momento. Aquí el robot es real o Gazebo/YARP puro: no existe ningún
sistema de visión o motion-capture que reporte la pose del objeto, así que
`lerobot_robot_icubrg.iCub` (este paquete) **no** expone esas features — ni
siquiera como campos vacíos. El resto del contrato de observación/acción
(estado articular, velocidad/torque/wrench/EEF-pose/tactile opcionales,
cámaras, y las acciones cartesianas por mano) es el mismo en ambos modos.
Ver [`lerobot-robot-icubrg/README.md`](common/lerobot-robot-icubrg/README.md).

## Instalación

```bash
cd common/lerobot-robot-icubrg && pip install .
cd ../lerobot-teleoperator-icubrgteleop && pip install .
```

O usa `environment.yml` para crear el entorno conda completo (incluye YARP,
icub-main, MuJoCo y ambos plugins en modo editable):

```bash
conda env create -f environment.yml
conda activate icub_lerobot_gazebo_env
```

## Configuración

`config/control_config.yaml` es compartido por los tres componentes
(`teleop_module_sm.py`, `lerobot_robot_icubrg.iCub` y
`lerobot_teleoperator_icubrgteleop.iCubTeleop`). Los campos relevantes:

- `robot_name`: `"icub"` (real) o `"icubSim"` (Gazebo) — selecciona los
  puertos de cámara y qué instancia YARP se usa.
- `control_arms`: `left`, `right`, `both` — sobreescrito por
  `--control-arms auto` si detecta menos brazos disponibles.
- `gaze_ctrl` / `cart_ctrl`: activan el Gaze Controller y el Cartesian
  Controller en `icubyarpinterface.py`.
- `tactile_*`: republicado por `teleop_module_sm.py` en `/teleop/touch:o`
  (no usado por `play_gazebo.py` salvo que se active `use_tactile` en el
  robot; es la vía usada por herramientas de grabación de más bajo nivel).

También puedes fijar las rutas por variable de entorno en vez de `--config`/`--model`:

```bash
export ICUB_LEROBOT_CONFIG=/home/icub/mujoco_ws/REPO_ICUB/icub-lerobot-gazebo/config/control_config.yaml
export ICUB_MUJOCO_MODEL_PATH=/home/icub/mujoco_ws/REPO_ICUB/icub-lerobot-gazebo/mujoco/assets/scenes/scene_icub_empty_table.xml
```

## Preparación física para la teleoperación

1. Siéntate cómodamente en una silla.
2. Coloca ambos brazos en forma de "L" (como si fueras a comer en una mesa).
3. Mantén esta postura antes de correr `play_gazebo.py`: el sistema la usa
   como tu punto de origen. Las posiciones de las manos virtuales serán
   relativas a este punto.

## Estructura de esta carpeta

Organizada por responsabilidad: lo que sirve igual para robot real y simulado
va en `common/`, lo exclusivo de cada lado en `gazebo/` y `mujoco/`.

```
icub-lerobot-gazebo/
├── play_gazebo.py                      # Entry point (teleop + grabación LeRobot)
│
├── config/
│   └── control_config.yaml             # ÚNICO switch real/simulado (robot_name) + brazos, gaze, tactile
│
├── common/                             # Agnóstico: funciona igual con robot real o Gazebo
│   ├── yarp_module/                    # Lado servidor, corre pegado al robot
│   │   ├── teleop_module_sm.py         # YARP RFModule (control boards, cartesian, gaze, RPC, cámaras→VR)
│   │   └── icubyarpinterface.py        # Interfaz YARP de bajo nivel
│   ├── lerobot-robot-icubrg/             # Plugin LeRobot "Robot" — cliente YARP
│   └── lerobot-teleoperator-icubrgteleop/# Plugin LeRobot "Teleoperator" — lee targets de MuJoCo
│
├── vr/                                 # Meta Quest / BeaVR
│   └── vr_usb.py                       # Conexión por cable (adb reverse), usada por --vr-cable
│
├── gazebo/                             # SOLO simulación Gazebo
│   ├── worlds/icub_world.sdf           # Mundo (mesa + cubo azul)
│   ├── models/                         # Modelos SDF (+ symlink iCub → icub-models del conda env)
│   ├── configs/
│   └── scripts/                        # start_sim.sh, stop_sim.sh, reset_objects.sh
│
├── mujoco/                             # SOLO el visualizador
│   └── assets/{scenes,meshes,textures} # Escenas MJCF y las mallas/texturas que usan
│
└── _unused/                            # Apartado para revisión, no lo usa nadie (ver más abajo)
```

### `_unused/`

Código y assets que ninguna ruta de ejecución alcanza. Se apartaron en vez de
borrarse para que puedas confirmar antes de eliminarlos:

- `dependencies/teleop_mujoco.py` + `mujoco_ik.py` + `recorder_mujoco.py` +
  `zmq_image_publisher.py` — el visualizador MuJoCo standalone anterior a los
  plugins LeRobot. Nada lo importa, y publica imágenes de MuJoCo en el puerto
  10505, el mismo que `teleop_module_sm.py` usa para mandar las cámaras del
  robot al visor — contradice el flujo que quieres.
- `dependencies/dataset_curation.py` — sin importadores ni CLI.
- `assets/` — 155 mallas y 23 texturas que ninguna escena referencia (~94 MB),
  más `meshes/objaverse/`.

