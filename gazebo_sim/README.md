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

Es el que usa el Hub en *Configuración → 2.1 Start sim*. Crea en segundo plano una sesión de
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

## Lanzador en primer plano (`scripts/launch_sim.sh`)

Alternativa para usar sin el Hub: levanta `yarpserver` (si no hay uno) y `gz sim` con GUI en la
misma terminal, y permite elegir otros mundos y modelos (`--world empty --model ...`, ver
`--help`). Con Ctrl+C cierra todo.

## Escenas y objetos

- Tareas: `mujoco_sim/assets/scenes/scenes.yaml`, el mismo manifiesto de MuJoCo.
- `scripts/reset_objects.sh` crea los objetos de la tarea elegida (`$ICUB_SCENE_FILE`, lo escribe
  el Hub) en posiciones al azar dentro de los rangos de `scenes.yaml`, pasadas al marco de Gazebo
  ((x, y) -> (-x, -y)). Lo llaman el Hub, la tecla `n` / el botón Y del VR y `eval_policy.py`.
- `scene_fixtures.yaml`: piezas fijas de cada tarea (la puerta).
- `models/`: modelos de los objetos, generados desde MuJoCo con `src/tools/object_sdf_from_mujoco.py`
  y `src/tools/door_sdf_from_mujoco.py` (mismas medidas, masas y fricciones); `top-camera-stand`
  (cámara cenital) y la cámara VR del iCub (`iCubGazeboV2_5_visuomanip_camada/vr_camera_nws.xml`).
