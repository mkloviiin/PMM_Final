import time
import numpy as np
import yarp
from pyquaternion import Quaternion

# Tiempo máximo que se espera a que un puerto del robot aparezca. Gazebo levanta
# yarprobotinterface / iKinCartesianSolver / iKinGazeCtrl 5-15 s después de
# arrancar: sin esta espera el módulo moría si se lanzaba antes de tiempo.
PORT_WAIT_TIMEOUT_S = 60.0


def _port_responds(port_name, timeout=5.0):
    """True si el puerto responde a `yarp ping` dentro del timeout.

    Va en un subproceso porque, si el servidor está colgado, la conexión YARP desde
    Python se bloquea sin respetar timeouts (y con ella todo el módulo).
    """
    import subprocess
    import sys
    from pathlib import Path
    yarp_bin = Path(sys.executable).parent / "yarp"
    try:
        r = subprocess.run([str(yarp_bin) if yarp_bin.exists() else "yarp", "ping", port_name],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=timeout)
        return r.returncode == 0 and "This is" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return True  # sin `yarp` CLI no se puede comprobar: no bloquear el arranque


def _ping_output(port_name, timeout=5.0):
    """Salida de `yarp ping <port>` (subproceso con timeout), o None si no responde."""
    import subprocess
    import sys
    from pathlib import Path
    yarp_bin = Path(sys.executable).parent / "yarp"
    try:
        r = subprocess.run([str(yarp_bin) if yarp_bin.exists() else "yarp", "ping", port_name],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           timeout=timeout)
        return r.stdout if "This is" in r.stdout else None
    except Exception:
        return None


def _wait_for_port(port_name, timeout=PORT_WAIT_TIMEOUT_S):
    """Espera a que `port_name` esté registrado en el yarpserver. True si aparece.

    Solo consulta el name server: Network.exists() además se conecta al puerto y, si el
    proceso dueño está colgado, se bloquea para siempre sin imprimir nada. Si el puerto
    responde o no lo comprueba _port_responds (con timeout) donde haga falta.
    """
    deadline = time.time() + timeout
    warned = False
    while not yarp.Network.queryName(port_name).isValid():
        if time.time() >= deadline:
            print(f"  ✗ Timeout ({timeout:.0f} s) waiting for {port_name}")
            return False
        if not warned:
            print(f"  ⏳ Waiting for {port_name} (robot/sim still starting?)...")
            warned = True
        time.sleep(1.0)
    return True

class iCubYARPInterface:
    """Encapsula toda la comunicación con el robot iCub (real o simulado) vía YARP."""
    SHOULDER_YAW_MIN_DEG = -22.0
    SHOULDER_YAW_MAX_DEG = 70.0
    
    def __init__(self, 
                 robot_name="icub", 
                 actuators_dict=None,
                 cartesian_arms=None, 
                 enable_cartesian=True,
                 primary_arm_for_torso="right_arm",
                 home_pose_deg=None,
                 use_gaze=False,
                 bind_eyes=True,
                 cart_carrier="udp"):
        """
        Args:
            robot_name (str): Nombre del robot.
            actuators_dict (dict): Diccionario de partes y articulaciones.
            cartesian_arms (list): Lista de brazos a controlar.
            enable_cartesian (bool): Habilita la interfaz cartesiana.
            primary_arm_for_torso (str): Brazo que controlará el torso.
            home_pose_deg (dict): Diccionario con la pose home {joint: deg}.
            cart_carrier (str): Carrier de los puertos de streaming del cliente
                cartesiano (command/state/events). "udp" es el default de iCub.
        """
        
        yarp.Network.init()
        self.robot_name = robot_name
        
        if actuators_dict is None:
            raise ValueError("Se requiere un 'actuators_dict' para la inicialización.")
            
        self.parts = list(actuators_dict.keys())
        self.actuators_to_control = [joint for part_joints in actuators_dict.values() 
                                                for joint in part_joints]

        # Guardar home pose como variable global de la clase
        self.home_pos_deg = home_pose_deg if home_pose_deg is not None else {}

        self.drivers = {}
        self.encoders = {}
        self.pos_direct = {}
        self.pos_control = {}
        self.control_modes = {}
        # Último modo puesto en cada brazo por _set_arm_control_mode. Cualquier otro
        # cambio de modo (home, start_teleop, init) lo invalida con pop().
        self._arm_mode: dict[str, str] = {}
        # Si es False (por defecto) el módulo NO toca los modos del brazo en modo
        # cartesiano: el cartesianController elige el suyo (POSITION_DIRECT o VELOCITY
        # según su config). Forzar POSITION_DIRECT + stop() dejaba quieto el brazo del
        # robot real. True = comportamiento antiguo (cart_force_position_direct en el YAML).
        self.cart_force_position_direct = False
        # Diagnóstico: segundos gastados por llamada al robot en el ciclo actual
        # (cycle_times, lo resetea el módulo en cada ciclo) y por ventana de reporte
        # (window_times: nombre -> [n, total, max]); frames de cámara recibidos por ojo.
        self.cycle_times: dict[str, float] = {}
        self.window_times: dict[str, list] = {}
        self.cam_frames = {"left": 0, "right": 0}
        self.axis_info = {}
        self.num_joints_part = {}
        self.joint_processing_map = []

        print("Connecting to YARP remote control boards...")
        for part in self.parts:
            self._open_driver(part)
        print("YARP connection successful.")

        self._map_joints()

        # --- Configuración Cartesiana ---
        self.cart_drivers = {}
        self.cart_interfaces = {}
        self.enable_cartesian = enable_cartesian
        self.cart_carrier = cart_carrier
        
        if not self.enable_cartesian:
            self.cart_parts = []
        elif cartesian_arms is None:
            self.cart_parts = ["left_arm", "right_arm"]
        else:
            self.cart_parts = list(cartesian_arms)

        if self.cart_parts:
            print(f"Initializing Cartesian controllers for: {self.cart_parts}")
            for part in self.cart_parts:
                self._open_cartesian_driver(part)
            print("YARP Cartesian connection successful.")
        else:
            print("Cartesian control disabled or no active arms. Skipping Cartesian drivers.")
        
        # --- Configurar Prioridad Inicial del Torso ---
        self.primary_arm = primary_arm_for_torso
        if self.primary_arm in self.cart_parts:
            self.set_primary_arm_for_torso(self.primary_arm)
        elif self.cart_parts:
            new_primary = self.cart_parts[0]
            print(f"Warning: Primary arm '{self.primary_arm}' not active. Setting to '{new_primary}'.")
            self.primary_arm = new_primary 
            self.set_primary_arm_for_torso(self.primary_arm)

        # --- Configuración de Cámaras ---
        self.cam_parts = ["left", "right"]
        self.cam_ports = {}
        self.img_buffers = {}
        self.np_arrays = {}
        print("Connecting to YARP Camera ports...")
        for cam in self.cam_parts:
            self._open_camera_driver(cam)

        # --- Configuración Gaze Controller ---
        self.gaze_driver = None
        self.igaze = None
        if use_gaze:
            if not self.init_gaze_controller(bind_eyes=bind_eyes):
                raise RuntimeError(f"Failed to connect to Gaze Controller")

    # --- Gestión Unificada de Home Pose ---
    def go_to_home_pose(self, wait=False):
        """
        Mueve el robot a la posición 'home' almacenada en self.home_pos_deg.
        
        Args:
            wait (bool): Si es True, bloquea hasta terminar. 
                         Si es False, envía el comando y retorna inmediatamente.
        """
        if not self.home_pos_deg:
            print("Warning: No home pose defined in configuration.")
            return

        print(f"Moving to home pose (Blocking={wait})...")
        
        for part in self.parts:
            num_j = self.num_joints_part[part]
            part_pos_vec = yarp.Vector(num_j)
            
            # 1. Cambiar modo a POSITION
            modes = yarp.VectorInt(num_j, yarp.VOCAB_CM_POSITION)
            self.control_modes[part].setControlModes(modes.data())
            self._arm_mode.pop(part, None)

            # 2. Preparar vector con valores home o actuales (si no está en home dict)
            # Primero leemos encoders actuales para llenar huecos
            current_encs = yarp.Vector(num_j)
            self.encoders[part].getEncoders(current_encs.data())
            
            for j in range(num_j):
                part_pos_vec[j] = current_encs[j] # Default al actual
                joint_name = self.axis_info[part].getAxisName(j)
                if joint_name in self.home_pos_deg:
                    part_pos_vec[j] = self.home_pos_deg[joint_name] # Sobrescribir con home

            if self.igaze is not None and part == "head":
                continue

            ipos = self.drivers[part].viewIPositionControl()

            # 3. Enviar comando
            ipos.positionMove(part_pos_vec.data())
        
        # Si se pide bloquear, esperamos aquí
        if wait:
            print("Waiting for home motion to complete...")
            while not self.check_motion_done():
                time.sleep(0.1)
            print("Home pose reached.")
            self.set_all_parts_to_position_direct() # Restaurar control directo al terminar

    def check_motion_done(self):
        """
        Verifica si todas las partes controladas han terminado su movimiento (IPositionControl).
        Retorna True si está quieto, False si se mueve.
        """
        all_done = True
        for part in self.parts:
            ipos = self.drivers[part].viewIPositionControl()
            if not ipos.checkMotionDone():
                all_done = False
                break
        return all_done
    
    def set_all_parts_to_position_direct(self):
        """Restaura el modo POSITION_DIRECT para control rápido."""
        print("Restoring control modes to POSITION_DIRECT...")
        for part in self.parts:

            if self.igaze is not None and part == "head":
                continue
            
            num_j = self.num_joints_part[part]
            modes = yarp.VectorInt(num_j)
            for i in range(num_j):
                if i < 7: # Brazo/Torso
                    modes[i] = yarp.VOCAB_CM_POSITION
                else: # Manos se quedan en POSITION
                    modes[i] = yarp.VOCAB_CM_POSITION
            self.control_modes[part].setControlModes(modes.data())
            self._arm_mode.pop(part, None)
    # ------------------------------------------------------

    def get_controllable_joints(self):
        return self.actuators_to_control

    def _open_driver(self, part_name):
        props = yarp.Property()
        props.put("device", "remote_controlboard")
        props.put("local", f"/mujoco_sync/{part_name}")
        props.put("remote", f"/{self.robot_name}/{part_name}")
        _wait_for_port(f"/{self.robot_name}/{part_name}/rpc:i")

        driver = yarp.PolyDriver(props)
        if not driver.isValid():
            raise RuntimeError(f"Failed to connect to {part_name}")
            
        self.drivers[part_name] = driver
        self.encoders[part_name] = driver.viewIEncoders()
        self.pos_direct[part_name] = driver.viewIPositionDirect()
        self.pos_control[part_name] = driver.viewIPositionControl()
        self.control_modes[part_name] = driver.viewIControlMode()
        self.axis_info[part_name] = driver.viewIAxisInfo()
        control_limits = driver.viewIControlLimits()
        
        num_axes = self.encoders[part_name].getAxes()
        self.num_joints_part[part_name] = num_axes
        
        # CRITICO: Leer encoders ANTES de cualquier cambio de modo
        current_encs = yarp.Vector(num_axes)
        self.encoders[part_name].getEncoders(current_encs.data())
        if part_name == "head":
            print(f"  [HEAD] Posición actual: [{current_encs[0]:.1f}, {current_encs[1]:.1f}, {current_encs[2]:.1f}]")
        
        # Inicializar en Position (no debería mover nada porque es el modo por defecto)
        modes = yarp.VectorInt(num_axes, yarp.VOCAB_CM_POSITION)
        self.control_modes[part_name].setControlModes(modes.data())

        # Aplicar límites explícitos de shoulder yaw en ambos brazos
        if control_limits is not None and part_name in ("right_arm", "left_arm"):
            for joint_idx in range(num_axes):
                joint_name = self.axis_info[part_name].getAxisName(joint_idx)
                if "shoulder_yaw" in joint_name:
                    control_limits.setLimits(
                        joint_idx,
                        self.SHOULDER_YAW_MIN_DEG,
                        self.SHOULDER_YAW_MAX_DEG,
                    )
                    print(
                        f"  [{part_name}] Limit set for {joint_name}: "
                        f"[{self.SHOULDER_YAW_MIN_DEG}, {self.SHOULDER_YAW_MAX_DEG}] deg"
                    )
        
        # Para la cabeza, escribir inmediatamente los valores actuales para mantener posición
        if part_name == "head":
            ipos = driver.viewIPositionControl()
            ipos.positionMove(current_encs.data())

    def _map_joints(self):
        main_idx = 0
        for part_name in self.parts:
            num_axes = self.num_joints_part[part_name]
            for yarp_idx in range(num_axes):
                joint_name = self.axis_info[part_name].getAxisName(yarp_idx)
                if joint_name in self.actuators_to_control:
                    control_idx = self.actuators_to_control.index(joint_name)
                    self.joint_processing_map.append(
                        (part_name, joint_name, yarp_idx, control_idx, main_idx)
                    )
                    main_idx += 1

    def get_joint_state(self):
        q_rad = np.zeros(len(self.actuators_to_control))
        dq_rad = np.zeros(len(self.actuators_to_control))
        raw_positions = {}
        raw_velocities = {}
        
        for part in self.parts:
            num_j = self.num_joints_part[part]
            pos_vec = yarp.Vector(num_j)
            vel_vec = yarp.Vector(num_j)
            self.encoders[part].getEncoders(pos_vec.data())
            self.encoders[part].getEncoderSpeeds(vel_vec.data())
            raw_positions[part] = [pos_vec[j] for j in range(num_j)]
            raw_velocities[part] = [vel_vec[j] for j in range(num_j)]

        for part_name, joint_name, yarp_idx, control_idx, main_idx in self.joint_processing_map:
            q_rad[control_idx] = np.deg2rad(raw_positions[part_name][yarp_idx])
            dq_rad[control_idx] = np.deg2rad(raw_velocities[part_name][yarp_idx])
        return q_rad, dq_rad

    def send_hand_positions(self, arm, target_positions_deg):
        if arm not in self.pos_control or self.pos_control[arm] is None: return False
        num_total = self.num_joints_part[arm]
        num_hand = num_total - 7
        if len(target_positions_deg) != num_hand: return False

        ipos = self.pos_control[arm]
        
        # Asegurar modo POSITION para la mano
        modes = yarp.VectorInt(num_total)
        self.control_modes[arm].getControlModes(modes.data())
        for i in range(7, num_total):
            if modes[i] != yarp.VOCAB_CM_POSITION:
                 self.control_modes[arm].setControlMode(i, yarp.VOCAB_CM_POSITION)

        for i in range(num_hand):
            joint_index = i + 7
            ipos.positionMove(joint_index, float(target_positions_deg[i]))
        return True

    def get_hand_positions(self, arm):
        if arm not in self.encoders: return None
        num_total = self.num_joints_part[arm]
        num_hand = num_total - 7
        all_encs = yarp.Vector(num_total)
        self.encoders[arm].getEncoders(all_encs.data())
        hand_pos = np.zeros(num_hand)
        for i in range(num_hand):
            hand_pos[i] = all_encs[i + 7]
        return hand_pos

    def set_eyes_positions(self, eyes_tilt_deg, eyes_version_deg, eyes_vergence_deg):
        num_j = self.num_joints_part["head"]

        eyes_indices = {}
        for i in range(num_j):
            joint_name = self.axis_info["head"].getAxisName(i)
            if "eye" in joint_name.lower():
                if "tilt" in joint_name.lower():
                    eyes_indices["tilt"] = i
                elif "version" in joint_name.lower():
                    eyes_indices["version"] = i
                elif "vergence" in joint_name.lower():
                    eyes_indices["vergence"] = i

        # Ensure that the eyes are in POSITION_DIRECT
        for idx in eyes_indices.values():
            current_mode = self.control_modes["head"].getControlMode(idx)
            if current_mode != yarp.VOCAB_CM_POSITION_DIRECT:
                self.control_modes["head"].setControlMode(idx, yarp.VOCAB_CM_POSITION_DIRECT)

        ipos_direct = self.pos_direct["head"]
        ipos_direct.setPosition(eyes_indices["tilt"], eyes_tilt_deg)
        ipos_direct.setPosition(eyes_indices["version"], eyes_version_deg)
        ipos_direct.setPosition(eyes_indices["vergence"], eyes_vergence_deg)

    def _open_cartesian_driver(self, arm_part):
        props = yarp.Property()
        props.put('device', 'cartesiancontrollerclient')
        props.put('local', f'/cartesian_client/{arm_part}')
        props.put('remote', f'/{self.robot_name}/cartesianController/{arm_part}')
        props.put('timeout', 60.0)
        # goToPose no espera respuesta: va por command:o con este carrier. Con udp, en
        # la red del robot real los comandos se pierden y el brazo no se mueve.
        props.put('carrier', self.cart_carrier)
        rpc_port = f'/{self.robot_name}/cartesianController/{arm_part}/rpc:i'
        _wait_for_port(rpc_port)
        if not _port_responds(rpc_port):
            raise RuntimeError(
                f"Cartesian controller {rpc_port} is registered but NOT responding (hung). "
                f"Restart the robot's yarprobotinterface (it hosts cartesianController/{arm_part}) "
                f"with iKinCartesianSolver --part {arm_part} running, then start the module again.")
        # El puerto del controlador aparece antes de que su solver esté listo
        # ("unable to connect to solver!"): se reintenta hasta el timeout.
        deadline = time.time() + PORT_WAIT_TIMEOUT_S
        driver = yarp.PolyDriver(props)
        while not driver.isValid() and time.time() < deadline:
            print(f"  ⏳ Cartesian controller {arm_part} not ready yet, retrying...")
            time.sleep(2.0)
            driver = yarp.PolyDriver(props)
        if not driver.isValid(): raise RuntimeError(f"Failed cartesians {arm_part}")
        # Sin el enlace controlador -> solver el controlador acepta goToPose pero nunca
        # mueve el brazo (no recibe soluciones). Solo avisa: la cabeza/cámaras siguen.
        ping = _ping_output(f"/cartesianSolver/{arm_part}/in")
        if ping is not None and f"cartesianController/{arm_part}" not in ping:
            print("  " + "!" * 70)
            print(f"  ✗ /cartesianSolver/{arm_part} is NOT linked to "
                  f"/{self.robot_name}/cartesianController/{arm_part}: the arm will NOT move.")
            print(f"    Restart on the robot, in this order: yarprobotinterface, then "
                  f"iKinCartesianSolver --part {arm_part}; then restart this module.")
            print("  " + "!" * 70)
        elif ping is not None:
            print(f"  ✓ Cartesian solver linked for {arm_part}")
        self.cart_drivers[arm_part] = driver
        self.cart_interfaces[arm_part] = driver.viewICartesianControl()
        self.cart_interfaces[arm_part].setTrajTime(2.0)
        # NOTA: NO llamar set_head_to_position_direct() aquí
        # Eso se hace una sola vez después de toda la inicialización

    def look_at_sync(self, x, y, z):
        if self.igaze is None: return
        target = yarp.Vector(3)
        target.set(0, x); target.set(1, y); target.set(2, z)
        self.igaze.lookAtFixationPointSync(target)
        self.set_head_to_position_direct()


    def _get_current_pose(self, arm):
        if arm not in self.cart_interfaces: return None, None
        pos = yarp.Vector(3)
        ax = yarp.Vector(4)
        if not self.cart_interfaces[arm].getPose(pos, ax): return None, None
        
        curr_pos = np.array([pos[i] for i in range(3)]) # [mts]
        axis = np.array([ax[0], ax[1], ax[2]]) # [mts]
        angle = np.array(ax[3])   # [rads]
        q = Quaternion(axis=axis, radians=angle)
        return curr_pos, np.array([q.w, q.x, q.y, q.z])

    def _check_pose_threshold(self, c_pos, c_quat, t_pos, t_quat, th_p, th_o):
        if c_pos is None: return False
        p_err = np.linalg.norm(c_pos - t_pos)
        q_c = Quaternion(c_quat)
        q_t = Quaternion(t_quat)
        q_err = q_t * q_c.inverse
        a_err = np.rad2deg(q_err.angle)
        if a_err > 180: a_err = 360 - a_err
        return p_err <= th_p # and a_err <= th_o

    def set_primary_arm_for_torso(self, arm):
        if arm not in self.cart_interfaces: return
        self.primary_arm = arm
        other = "left_arm" if arm == "right_arm" else "right_arm"
        
        # Enable torso for primary
        en = yarp.Vector(3); en.set(0,1); en.set(1,2); en.set(2,1)
        dof = yarp.Vector()
        self.cart_interfaces[arm].getDOF(dof)
        self.cart_interfaces[arm].setDOF(en, dof)
        #LIMITES TORSO
        self.cart_interfaces[arm].setLimits(0,-15,15) # pitch
        self.cart_interfaces[arm].setLimits(1,-0, 0) # roll
        self.cart_interfaces[arm].setLimits(2,-10,10) # pitch

        # Disable torso for other
        if other in self.cart_interfaces:
            dis = yarp.Vector(3); dis.zero()
            self.cart_interfaces[other].getDOF(dof)
            self.cart_interfaces[other].setDOF(dis, dof)

    def go_to_pose_async(self, arm, pos, quat_wxyz):
        if arm not in self.cart_interfaces: return False
        # pos/quat ya llegan en el frame root del robot (el teleoperador los convierte)
        t_pos = np.asarray(pos, dtype=float)

        c_pos, c_quat = self._timed("getPose", self._get_current_pose, arm)
        # CHANGED: Threshold increased from 0.01 to 0.025 to reduce micro-corrections
        if self._check_pose_threshold(c_pos, c_quat, t_pos, quat_wxyz, 0.01, 5.0):
            return True

        pos_y = yarp.Vector(3)
        pos_y.set(0, t_pos[0]); pos_y.set(1, t_pos[1]); pos_y.set(2, t_pos[2])
        q = Quaternion(quat_wxyz)
        ax_y = yarp.Vector(4)
        # goToPose espera eje-angulo con theta en RADIANES (igual que getPose lo devuelve)
        ax_y[0]=q.axis[0]; ax_y[1]=q.axis[1]; ax_y[2]=q.axis[2]; ax_y[3]=q.angle
        
        self._timed("set_mode", self._set_arm_control_mode, arm, 'cartesian')
        return self._timed("goToPose", self.cart_interfaces[arm].goToPose, pos_y, ax_y)

    def _timed(self, name, fn, *args):
        """Llama fn(*args) y suma su duración a cycle_times/window_times[name]."""
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            dt = time.perf_counter() - t0
            self.cycle_times[name] = self.cycle_times.get(name, 0.0) + dt
            w = self.window_times.setdefault(name, [0, 0.0, 0.0])
            w[0] += 1
            w[1] += dt
            w[2] = max(w[2], dt)

    def _set_arm_control_mode(self, arm, mode):
        if arm not in self.control_modes: return
        # Solo al cambiar de modo: se llama en cada target (~30 Hz) y tanto stop()
        # como setControlModes son RPC al robot. Repetirlos frenaba el loop en el
        # robot real y el stop() cortaba el open/close de la mano a medio camino.
        if self._arm_mode.get(arm) == mode: return
        self._arm_mode[arm] = mode
        if mode == 'cartesian' and not self.cart_force_position_direct:
            return
        # Simplificado: Si es cartesian o direct, usamos POSITION_DIRECT
        y_mode = yarp.VOCAB_CM_POSITION_DIRECT if mode in ['cartesian', 'position_direct'] else yarp.VOCAB_CM_POSITION
        
        if mode == 'cartesian' and arm in self.drivers:
             self.drivers[arm].viewIPositionControl().stop()
             
        num = self.num_joints_part[arm]
        modes = yarp.VectorInt(num)
        for i in range(num):
            modes[i] = y_mode if i < 7 else yarp.VOCAB_CM_POSITION
        self.control_modes[arm].setControlModes(modes.data())

    def stop_arm_control(self, arm):
        if arm in self.cart_interfaces: self.cart_interfaces[arm].stopControl()
        if arm in self.drivers: self.drivers[arm].viewIPositionControl().stop()

    def check_shoulder_constraints(self, arm):
        """
        Verifica que la configuración actual del hombro esté dentro de los límites seguros.
        Basado en las restricciones del iCub para evitar singularidades y poses peligrosas.
        
        Returns:
            bool: True si la pose es segura, False si viola restricciones.
        """
        if arm not in self.encoders:
            return True  # Si no hay encoder, asumimos OK
        
        num_j = self.num_joints_part[arm]
        encs = yarp.Vector(num_j)
        self.encoders[arm].getEncoders(encs.data())
        
        # Joints del hombro (índices 0, 1, 2 en el brazo)
        shoulder_pitch = encs[0]  # r/l_shoulder_pitch
        shoulder_roll = encs[1]   # r/l_shoulder_roll  
        shoulder_yaw = encs[2]    # r/l_shoulder_yaw
        
        # Matriz de restricciones del hombro iCub (de robotology docs)
        # Estas son restricciones acopladas entre las 3 articulaciones del hombro
        c = 1.71
        A = np.array([[c, -c, 0],
                      [c, -c, -c],
                      [0, 1, 1],
                      [-c, c, c],
                      [0, -1, -1]])
        b = np.array([347.0, 366.57, 66.6, 112.42, 213.3])
        
        x = np.array([shoulder_pitch, shoulder_roll, shoulder_yaw])
        result = np.matmul(A, x) + b
        yaw_in_range = self.SHOULDER_YAW_MIN_DEG <= shoulder_yaw <= self.SHOULDER_YAW_MAX_DEG
        
        is_safe = bool(np.all(result > 0)) and yaw_in_range
        
        if not is_safe:
            print(f"[WARNING] Shoulder constraint violation on {arm}!")
            print(f"  Pitch={shoulder_pitch:.1f}, Roll={shoulder_roll:.1f}, Yaw={shoulder_yaw:.1f}")
            print(f"  Constraint values: {result}")
            if not yaw_in_range:
                print(
                    f"  Yaw out of bounds: [{self.SHOULDER_YAW_MIN_DEG}, "
                    f"{self.SHOULDER_YAW_MAX_DEG}]"
                )
        
        return is_safe

    def get_shoulder_safety_margin(self, arm):
        """
        Retorna el margen de seguridad mínimo de las restricciones del hombro.
        Valores negativos indican violación de restricciones.
        """
        if arm not in self.encoders:
            return float('inf')
        
        num_j = self.num_joints_part[arm]
        encs = yarp.Vector(num_j)
        self.encoders[arm].getEncoders(encs.data())
        
        c = 1.71
        A = np.array([[c, -c, 0],
                      [c, -c, -c],
                      [0, 1, 1],
                      [-c, c, c],
                      [0, -1, -1]])
        b = np.array([347.0, 366.57, 66.6, 112.42, 213.3])
        
        x = np.array([encs[0], encs[1], encs[2]])
        result = np.matmul(A, x) + b
        yaw_margin = min(
            encs[2] - self.SHOULDER_YAW_MIN_DEG,
            self.SHOULDER_YAW_MAX_DEG - encs[2],
        )
        
        return float(min(np.min(result), yaw_margin))

    def get_camera_images(self):
        l, r = None, None
        for eye in ["left", "right"]:
            if eye in self.cam_ports and (img := self.cam_ports[eye].read(False)):
                w = img.width()
                h = img.height()
                
                # Check if buffer needs resizing
                if shape := self.np_arrays[eye].shape[:2] != (h, w):
                    # Reallocate buffer
                    self.np_arrays[eye] = np.zeros((h, w, 3), dtype=np.uint8)
                    self.img_buffers[eye].resize(w, h)
                    self.img_buffers[eye].setExternal(
                        self.np_arrays[eye].data, 
                        self.np_arrays[eye].shape[1], 
                        self.np_arrays[eye].shape[0]
                    )

                self.img_buffers[eye].copy(img)
                self.cam_frames[eye] += 1

                if eye == "left":
                    l = self.np_arrays[eye]
                else:
                    r = self.np_arrays[eye]
        
        # --- Codigo original para referencia ---
        # if "left" in self.cam_ports and (img := self.cam_ports["left"].read(False)):
        #     self.img_buffers["left"].copy(img)
        #     assert self.np_arrays["left"].__array_interface__['data'][0] == self.img_buffers["left"].getRawImage().__int__()
        #     l = self.np_arrays["left"]
        # if "right" in self.cam_ports and (img := self.cam_ports["right"].read(False)):
        #     self.img_buffers["right"].copy(img)
        #     assert self.np_arrays["right"].__array_interface__['data'][0] == self.img_buffers["right"].getRawImage().__int__()
        #     r = self.np_arrays["right"]
        # ---------------------------------------
        return l, r

    def _open_camera_driver(self, cam, w=320, h=240):
        port = yarp.BufferedPortImageRgb()
        self.cam_ports[cam] = port
        p_name = f"/cam_client/{self.robot_name}/{cam}"
        port.open(p_name)
        robot_name_norm = self.robot_name.strip().lower()

        if robot_name_norm == "icub":
            profile_name = "REAL_ROBOT"
            source_port = f"/{self.robot_name}/camcalib/{cam}/out"
            source_label = "CALIBRATED"
        elif robot_name_norm == "icubsim":
            profile_name = "SIMULATION"
            source_port = f"/{self.robot_name}/cam/{cam}/rgbImage:o"
            source_label = "SIMULATION"
        else:
            raise ValueError(
                f"Unsupported robot_name='{self.robot_name}'. Expected 'icub' or 'icubSim'."
            )

        print(f"  [CAMERA] {cam}: using profile {profile_name} for robot_name='{self.robot_name}'")

        connected = _wait_for_port(source_port) and yarp.Network.connect(source_port, p_name, 'fast_tcp', False)
        if connected:
            print(f"  ✓ Camera {cam}: connected via {source_label} port ({source_port})")

        if not connected:
            raise ConnectionError(f"Failed to connect camera {cam}. Tried: {source_port}")
        
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        self.np_arrays[cam] = arr
        buf = yarp.ImageRgb()
        buf.resize(w,h)
        self.img_buffers[cam] = buf
        buf.setExternal(arr.data, arr.shape[1], arr.shape[0])

    # --- Inicializar Gaze Controller ---
    # En icubyarpinterface.py

    def init_gaze_controller(self, remote_port="/iKinGazeCtrl", bind_eyes=True):
        print("  [GAZE] Iniciando conexión al Gaze Controller...")
        _wait_for_port(f"{remote_port}/rpc")

        # 1. Pausar el controlador antes de conectar 
        
        rpc_port = yarp.RpcClient()
        rpc_port.open("/py/gaze_rpc_temp")
        if yarp.Network.connect("/py/gaze_rpc_temp", f"{remote_port}/rpc"):
            cmd = yarp.Bottle()
            reply = yarp.Bottle()
            
            # Detener
            cmd.addString("stop")
            rpc_port.write(cmd, reply)
            
            print("  [GAZE] >> restoreContext")
            cmd.clear()
            cmd.addString("restoreContext")
            rpc_port.write(cmd, reply)
            
            # C) Borrar cola de movimientos
            print("  [GAZE] >> clear")
            cmd.clear()
            cmd.addString("clear")
            rpc_port.write(cmd, reply)
            
            yarp.Network.disconnect("/py/gaze_rpc_temp", f"{remote_port}/rpc")
        rpc_port.close()
        
        time.sleep(0.2)
        
        # 2. Iniciar Driver Cliente
        props = yarp.Property()
        props.put("device", "gazecontrollerclient")
        props.put("remote", remote_port)
        props.put("local", "/py/gaze_client")
        
        deadline = time.time() + PORT_WAIT_TIMEOUT_S
        self.gaze_driver = yarp.PolyDriver(props)
        while not self.gaze_driver.isValid() and time.time() < deadline:
            print("  ⏳ Gaze Controller not ready yet, retrying...")
            time.sleep(2.0)
            self.gaze_driver = yarp.PolyDriver(props)
        if not self.gaze_driver.isValid():
            print(f"Error: No se pudo conectar al Gaze Controller en {remote_port}")
            return False

        self.igaze = self.gaze_driver.viewIGazeControl()
        
        # 3. Configuración de parámetros de suavizado
        #self.igaze.setEyesTrajTime(1.489)  # Optimized for eyes-down
        #self.igaze.setNeckTrajTime(1.512)   # Optimized for eyes-down
                
        # 4. Arrancar el controlador
        rpc_port = yarp.RpcClient()
        rpc_port.open("/py/gaze_rpc_run")
        if yarp.Network.connect("/py/gaze_rpc_run", f"{remote_port}/rpc"):
            print("  [GAZE] Enviando comando 'run'...")
            cmd = yarp.Bottle()
            reply = yarp.Bottle()
            cmd.addString("run")
            rpc_port.write(cmd, reply)  # Send RUN first!
            
            cmd.clear()
            cmd.addString("bind")
            cmd.addString("neck") 
            rpc_port.write(cmd, reply)

            if bind_eyes:
                cmd.clear()
                cmd.addString("bind")
                cmd.addString("eyes")
                rpc_port.write(cmd, reply)
            yarp.Network.disconnect("/py/gaze_rpc_run", f"{remote_port}/rpc")
            
        rpc_port.close()
        
        print("  [GAZE] Gaze Controller inicializado correctamente.")
        return True


    def look_at(self, x, y, z):
        """
        Mueve la cabeza/ojos para mirar a una coordenada 3D (en el marco del robot).
        """
        if self.igaze is None: return
        target = yarp.Vector(3)
        target.set(0, x); target.set(1, y); target.set(2, z)
        self.igaze.lookAtFixationPoint(target)

    def stop_gaze_controller(self):
        """Detiene el Gaze Controller y libera el control de la cabeza."""
        if self.gaze_driver and self.igaze:
            print("  [GAZE] Deteniendo controlador de mirada...")
            
            # Intentar usar RPC para stop/clear si es posible, o simplemente stop control
            # Usamos el path temporal que sabemos que funciona
            try:
                rpc_port = yarp.RpcClient()
                rpc_port.open("/py/gaze_rpc_stop")
                
                # Necesitamos saber el puerto remoto. Asumimos el default o guardamos
                # En init era /iKinGazeCtrl por default.
                remote_port = "/iKinGazeCtrl" 
                # (Mejor sería guardar remote_port en __init__, pero asumo default por ahora)
                
                if yarp.Network.connect("/py/gaze_rpc_stop", f"{remote_port}/rpc"):
                    cmd = yarp.Bottle()
                    reply = yarp.Bottle()
                    
                    cmd.addString("stop")
                    rpc_port.write(cmd, reply)
                    
                    cmd.clear()
                    cmd.addString("clear")
                    rpc_port.write(cmd, reply)
                    
                    yarp.Network.disconnect("/py/gaze_rpc_stop", f"{remote_port}/rpc")
                rpc_port.close()
            except Exception as e:
                print(f"  Warning: Error enviando stop RPC: {e}")

            # Importante: Liberar la referencia para que go_to_home_pose no salte la cabeza
            if self.gaze_driver:
                self.gaze_driver.close()
                self.gaze_driver = None
            self.igaze = None
            print("  [GAZE] Control liberado.")

    def set_head_to_position_direct(self):
        if "head" in self.control_modes:
            num_j = self.num_joints_part["head"]
            
            # 1. Leer encoders ACTUALES antes del cambio de modo
            current_encs = yarp.Vector(num_j)
            self.encoders["head"].getEncoders(current_encs.data())
            
            # 2. Cambiar modo a POSITION_DIRECT
            modes = yarp.VectorInt(num_j, yarp.VOCAB_CM_POSITION_DIRECT)
            self.control_modes["head"].setControlModes(modes.data())
            
            # 3. Escribir la posición actual inmediatamente para evitar salto
            # Esto asegura que el controlador mantenga la posición actual
            ipos_direct = self.pos_direct["head"]
            for i in range(num_j):
                ipos_direct.setPosition(i, current_encs[i])

    def close(self):
        for d in self.drivers.values(): d.close()
        for d in self.cart_drivers.values(): d.close()
        for p in self.cam_ports.values(): p.close()
        if self.gaze_driver and self.gaze_driver.isValid():
            self.gaze_driver.close()
        yarp.Network.fini()