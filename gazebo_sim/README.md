# Simulación del iCub en Gazebo y YARP

Este directorio contiene el mundo, los modelos y los scripts para lanzar y cerrar la simulación
del iCub en Gazebo Sim, con **YARP** para la comunicación y **tmux** para los procesos. Todo usa
el entorno Pixi del proyecto (`pixi install` en la raíz).

El robot lo controla el módulo YARP (`src/interfaces/teleop_module_sm.py`) con el solver
whole-body de BLF (IK o ID, ver `config/control_config.yaml`), conectado directamente a los
controlboards que abre el `RobotInterface` del modelo dentro de Gazebo. Ya no se usan
`yarprobotinterface`, `iKinCartesianSolver` ni `iKinGazeCtrl`.

---
## Script de inicio del Hub (`scripts/start_sim.sh`)

Es el que usa el Hub en *Configuración → 2.1 Start sim* (y la pestaña *5. Evaluate*). Siempre lanza
`iCubGazeboV2_5_visuomanip` en la mesa (`worlds/icub_world.sdf`). Crea en segundo plano una sesión de
`tmux` llamada `icub_sim` con:

1. **`server`**: `yarpserver --write`, el servidor de nombres de YARP.
2. **`gazebo`**: `gz sim -s -r worlds/icub_world.sdf` (servidor: física y sensores) y, en otro
   panel, `gz sim -g` (la GUI). Los logs quedan en `/tmp/tmux_output_pipe_gz_server` y
   `/tmp/tmux_output_pipe_gz_gui`.

```bash
bash gazebo_sim/scripts/start_sim.sh
tmux attach -t icub_sim        # ver las salidas (Ctrl+b w: ventanas, Ctrl+b d: salir)
bash gazebo_sim/scripts/stop_sim.sh
```

`GZ_RENDER_ENV=""` desactiva las variables de render con la GPU NVIDIA de un notebook con GPU
híbrida (PRIME), si no hacen falta.

`gz sim` (servidor y GUI) corre envuelto en `scripts/tmux_guard.sh`. Así Gazebo se cierra también
si la sesión tmux se cierra sin `stop_sim.sh`: por su cuenta ignora el SIGHUP del panel y quedaba
huérfano.

## Lanzador en primer plano (`scripts/launch_sim.sh`)

Alternativa desde la terminal: levanta `yarpserver` (si no hay uno) y `gz sim` con GUI en la misma
terminal. Con Ctrl+C cierra todo. Sin argumentos (`--world manip`) es la misma simulación que
`start_sim.sh`.

```bash
./gazebo_sim/scripts/launch_sim.sh                                          # tareas
./gazebo_sim/scripts/launch_sim.sh --world empty --model iCubGazeboV2_5_fixed   # depuración
./gazebo_sim/scripts/launch_sim.sh --stop                                   # mata gz sim y yarpserver
```

## Modelos del iCub

| Mundo (`worlds/`) | Modelo | Uso |
|---|---|---|
| `icub_world.sdf` (`--world manip`) | `iCubGazeboV2_5_visuomanip` (manos, ojos y cámaras), en la mesa | **Tareas, grabación y evaluación.** El único que usan el Hub, `start_sim.sh` y `reset_objects.sh` |
| `icub_visuomanip_fixed.sdf` | `iCubGazeboV2_5_visuomanip`, fijo en el aire | Depuración |
| `icub_fixed.sdf` | `iCubGazeboV2_5_fixed`, fijo en el aire | Depuración (**sin ojos ni manos**) |
| `icub_v2_7_fixed.sdf` | `iCubGazeboV2_7_fixed`, fijo en el aire | Depuración (**sin ojos ni manos**) |
| `icub_v2_5_floating.sdf` | `iCubGazeboV2_5`, base flotante, de pie | Depuración (**sin ojos ni manos**) |
| `icub_v2_7_floating.sdf` | `iCubGazeboV2_7`, base flotante, de pie | Depuración (**sin ojos ni manos**) |

Los modelos de depuración sirven para comprobar que el control por torque o impedancia del TSID
funciona en Gazebo (`play.py --backend gazebo --controller id [--impedance] --no-record`). Solo se
lanzan con `launch_sim.sh --world empty --model ...`. No tienen mesa ni objetos, y sus mundos no se
llaman `default`, así que `reset_objects.sh` no funciona en ellos. Sin ojos ni manos tampoco hay
cámaras ni dedos: no sirven para grabar ni evaluar.

`launch_sim.sh` y `start_sim.sh` escriben el modelo en `/tmp/icub_gazebo_model`. De ahí lo leen el
módulo YARP y `play.py`, que cargan la cinemática de ese modelo y apagan las cámaras si no es
`visuomanip`. `play.py --robot-model` lo fuerza.

## Escenas y objetos

- Tareas: `mujoco_sim/assets/scenes/scenes.yaml`, el mismo manifiesto de MuJoCo.
- `scripts/reset_objects.sh` crea los objetos de la tarea elegida (`$ICUB_SCENE_FILE`, lo escribe
  el Hub) en posiciones al azar dentro de los rangos de `scenes.yaml`, pasadas al marco de Gazebo
  ((x, y) -> (-x, -y)). Lo llaman el Hub, la tecla `n` / el botón Y del VR y `eval_policy.py`.
- `scene_fixtures.yaml`: piezas fijas de cada tarea (la puerta).
- `models/`: modelos de los objetos, generados desde MuJoCo con `src/tools/object_sdf_from_mujoco.py`
  y `src/tools/door_sdf_from_mujoco.py` (mismas medidas, masas y fricciones); `top-camera-stand`
  (cámara cenital) y la cámara VR del iCub (`iCubGazeboV2_5_visuomanip_camada/vr_camera_nws.xml`).
