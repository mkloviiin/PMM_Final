# Simulación del iCub en Gazebo y YARP

Este directorio contiene los scripts necesarios para automatizar el lanzamiento y cierre de una simulación completa del robot iCub en Gazebo, utilizando **YARP** para la comunicación y **tmux** para gestionar los múltiples procesos.

---
## Script de Inicio (`start_sim.sh`)

Este script se encarga de configurar y lanzar todos los componentes de software necesarios para la simulación.

### ¿Qué hace?
El script crea una sesión de `tmux` llamada `icub_sim` en segundo plano. `tmux` es un multiplexor de terminal que permite que múltiples programas se ejecuten en ventanas separadas dentro de una única sesión. Esto organiza el entorno y facilita la depuración.

### ¿Cómo se ejecuta?
Para iniciar toda la simulación, navega al directorio de los scripts y ejecuta el archivo.

```bash
# Navega al directorio correcto
cd examples/gazebo/scripts/

# Otorga permisos de ejecución (solo la primera vez)
chmod +x start_sim.sh

# Ejecuta el script
./start_sim.sh
```

### Componentes Lanzados
El script abre seis ventanas dentro de la sesión de `tmux`, cada una con una función específica:

1.  **`server`**:
    * **Proceso**: `yarpserver --write`
    * **Función**: Inicia el servidor de nombres de YARP, que es el núcleo central para que todos los demás módulos de software puedan encontrarse y comunicarse entre sí.

2.  **`gazebo`**:
    * **Proceso**: `gz sim -r icub_world.sdf`
    * **Función**: Lanza el simulador Gazebo, cargando el mundo 3D y el modelo del robot iCub definidos en el archivo `icub_world.sdf`.

3.  **`robotinterface`**:
    * **Proceso**: `yarprobotinterface`
    * **Función**: Actúa como un puente fundamental entre el robot simulado en Gazebo y el ecosistema YARP. Expone las articulaciones, sensores y actuadores del robot como puertos YARP, permitiendo que otros programas los controlen.

4.  **`right_arm`**:
    * **Proceso**: `iKinCartesianSolver --part right_arm`
    * **Función**: Inicia el solver de cinemática inversa para el brazo derecho. Este módulo recibe comandos de posición cartesiana (x, y, z) para la mano y los convierte en los comandos de posición articular necesarios.

5.  **`left_arm`**:
    * **Proceso**: `iKinCartesianSolver --part left_arm`
    * **Función**: Hace lo mismo que el anterior, pero para el brazo izquierdo.

6.  **`gaze`**:
    * **Proceso**: `iKinGazeCtrl`
    * **Función**: Inicia el controlador de la mirada, que coordina los movimientos de los ojos y el cuello para que el robot pueda fijar su vista en un punto en el espacio.

---
## Interaccion con la simulación

Como los procesos se ejecutan en tmux, no los verás directamente en tu terminal. Para ver las salidas de cada componente (por ejemplo, para depurar), necesitas "adjuntarte" a la sesión.

* **Para entrar a la sesión:**
```bash
tmux attach -t icub_sim
```
* **Para navegar dentro de tmux:**
    
    * `Ctrl+b` y luego `w`: Abre arbol jerárquico con ventanas creadas
    * `Ctrl+b` y luego `d`: Para salir de tmux

---
## Script de cierre (`stop_sim.sh`)

Este script se encarga de cerrar de forma ordenada y segura todos los procesos iniciados por `start_sim.sh`.

### ¿Qué hace?

Envía comandos de terminación `(Ctrl+C)` a cada ventana de `tmux` en un orden específico para evitar procesos "zombis" o errores. Finalmente, cierra la sesión de `tmux`.

### ¿Como se ejecuta?

Desde el mismo directorio, ejecuta el script de cierre.

```bash
# Asegúrate de estar en examples/gazebo/scripts/

# Otorga permisos de ejecución (solo la primera vez)
chmod +x stop_sim.sh

# Ejecuta el script
./stop_sim.sh
```
