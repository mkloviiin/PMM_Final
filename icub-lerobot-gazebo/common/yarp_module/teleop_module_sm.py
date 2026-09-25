# File: teleop_module_sm.py
# Adapted from yarprfmodule.py with state machine pattern from superquadrics_module.py

import sys
import os
import time
import yarp
import numpy as np
from pathlib import Path
import cv2
import yaml
import subprocess

# --- Append paths ---
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent   # common/yarp_module -> icub-lerobot-gazebo
sys.path.append(str(_HERE))

# --- Import custom modules ---
from icubyarpinterface import iCubYARPInterface

CONFIG_FILE = os.getenv("ICUB_LEROBOT_CONFIG", str(_PROJECT_ROOT / "config" / "control_config.yaml"))

# Import ZMQ Transmitter for VR
try:
    sys.path.insert(0, str(_PROJECT_ROOT.parent / "beavr-bot" / "src"))
    from beavr.teleop.common.network.publisher import ZMQCompressedImageTransmitter
except ImportError:
    ZMQCompressedImageTransmitter = None


class TeleopModuleSM(yarp.RFModule):
    """
    iCub Teleoperation Module with State Machine pattern.
    
    Uses RPC commands via respond() method and state machine in updateModule().
    Based on superquadrics_module.py pattern from rl-icub-dexterous-manipulation.
    
    States:
        - idle: Default state, waiting for commands
        - going_home: Moving robot to home pose
        - teleop_active: Processing continuous teleop targets
        - opening_hand_right/left: Opening specified hand
        - closing_hand_right/left: Closing specified hand
        - looking_at: Moving gaze to target
        
    RPC Commands:
        - go_home: Go to home pose
        - start_teleop: Enable continuous target processing
        - stop_teleop: Stop continuous processing, return to idle
        - open_right / open_left: Open specified hand
        - close_right / close_left: Close specified hand
        - look_at x y z: Look at 3D point
        - start_gaze: Enable gaze controller
        - stop_gaze: Disable gaze controller
        - reset_scenario: Reset objects in scene
        - quit: Close module
    """

    def configure(self, rf):
        try:
            print("="*60)
            print("STARTING TELEOP MODULE (STATE MACHINE)")
            print("="*60)
            
            # === STATE MACHINE INITIALIZATION ===
            self.state = 'idle'
            self.pending_target_rh = None
            self.pending_target_lh = None
            self.pending_gaze_target = None
            self.next_state_after_action = 'idle'
            self._closed = False
            
            print(f"[1/9] Loading configuration from {CONFIG_FILE}...")
            with open(CONFIG_FILE, 'r') as f:
                self.cfg = yaml.safe_load(f)
            print("✓ Configuration loaded successfully")

            # --- Parámetros desde YAML ---
            print("[2/9] Parsing configuration parameters...")
            robot_name_yaml = self.cfg.get("robot_name", "icubSim")
            robot_name = rf.find("robot").asString() if rf.check("robot") else robot_name_yaml
            control_arms_yaml = self.cfg.get("control_arms", "both")
            control_arms = rf.find("control").asString() if rf.check("control") else control_arms_yaml
            primary_arm = self.cfg.get("primary_arm", "right_arm")
            self.cart_ctrl_enabled = self.cfg.get("cart_ctrl", True)
            
            self.period = self.cfg.get("period", 0.02)
            self.show_cameras = self.cfg.get("show_cameras", True)
            self.home_pos_deg = self.cfg.get("home_pose", {})
            hand_open_values = self.cfg.get("hand_open_values")
            hand_closed_values = self.cfg.get("hand_close_values")
            self.use_gaze = self.cfg.get("gaze_ctrl", False)
            self.use_tactile = self.cfg.get("tactile_enabled", True)
            # Derivados de robot_name para no cablear icub/icubSim en dos sitios.
            self.tactile_source_port = self.cfg.get(
                "tactile_source_port", f"/{robot_name}/skin/right_hand_comp")
            self.tactile_include_palm = self.cfg.get("tactile_include_palm", True)
            self.tactile_palm_source_port = self.cfg.get(
                "tactile_palm_source_port", f"/{robot_name}/skin/right_hand")
            self.tactile_auto_connect = self.cfg.get("tactile_auto_connect", True)
            
            # Cache para imágenes
            self.l_img = None
            self.r_img = None
            print(
                f"✓ Robot: {robot_name}, Control: {control_arms}, "
                f"Cartesian: {self.cart_ctrl_enabled}, Gaze: {self.use_gaze}"
            )

            # --- Filtrar actuadores según configuración ---
            print("[3/9] Setting up actuators...")
            full_actuators = self.cfg.get("actuators", {})
            self.actuators_to_use = {}
            
            self.actuators_to_use["torso"] = full_actuators["torso"]
            
            self.cartesian_arms = []
            if control_arms in ["left", "both"]:
                self.actuators_to_use["left_arm"] = full_actuators["left_arm"]
                if self.cart_ctrl_enabled:
                    self.cartesian_arms.append("left_arm")
            if control_arms in ["right", "both"]:
                self.actuators_to_use["right_arm"] = full_actuators["right_arm"]
                if self.cart_ctrl_enabled:
                    self.cartesian_arms.append("right_arm")
            if control_arms == "none":
                print("✓ Arms control disabled (none selected)")           

            self.actuators_to_use["head"] = full_actuators["head"]
            print(f"✓ Active Cartesian Arms: {self.cartesian_arms}")
            if not self.cart_ctrl_enabled:
                print("✓ Cartesian control disabled by config (cart_ctrl=false)")

            # --- Inicializar Interfaz ---
            print("[4/9] Initializing YARP interface...")
            self.yarp_interface = iCubYARPInterface(
                robot_name=robot_name,
                actuators_dict=self.actuators_to_use,
                cartesian_arms=self.cartesian_arms,
                enable_cartesian=self.cart_ctrl_enabled,
                primary_arm_for_torso=primary_arm,
                home_pose_deg=self.home_pos_deg,
                use_gaze=self.use_gaze,
                bind_eyes=False
            )
            print("✓ YARP interface initialized")

            # --- Movimiento inicial a Home ---
            print("[5/9] Moving to home pose (async)...")
            self.yarp_interface.go_to_home_pose(wait=False)
            self.state = 'going_home'
            print("✓ Home pose command sent")

            # === YARP RPC COMMAND PORT (State Machine Control) ===
            print("[6/9] Creating RPC command port...")
            self.cmd_port = yarp.Port()
            if not self.cmd_port.open("/teleop/rpc:i"):
                print("✗ ERROR: Failed to open /teleop/rpc:i")
                return False
            self.attach(self.cmd_port)  # Key: attach for respond() to work
            print("✓ /teleop/rpc:i created and attached")

            # === YARP DATA PORTS (for teleop_active state) ===
            print("[7/9] Creating data ports...")
            self.ports = {}
            
            if "right_arm" in self.cartesian_arms:
                self.ports["rh_target"] = yarp.BufferedPortBottle()
                if not self.ports["rh_target"].open("/teleop/rh_target:i"):
                    print("✗ ERROR: Failed to open /teleop/rh_target:i")
                    return False
                print("  ✓ /teleop/rh_target:i created")

                self.ports["rh_current_pose"] = yarp.BufferedPortBottle()
                if not self.ports["rh_current_pose"].open("/teleop/rh_current_pose:o"):
                    print("✗ ERROR: Failed to open /teleop/rh_current_pose:o")
                    return False
                print("  ✓ /teleop/rh_current_pose:o created")
            
            if "left_arm" in self.cartesian_arms:
                self.ports["lh_target"] = yarp.BufferedPortBottle()
                if not self.ports["lh_target"].open("/teleop/lh_target:i"):
                    print("✗ ERROR: Failed to open /teleop/lh_target:i")
                    return False
                print("  ✓ /teleop/lh_target:i created")

                self.ports["lh_current_pose"] = yarp.BufferedPortBottle()
                if not self.ports["lh_current_pose"].open("/teleop/lh_current_pose:o"):
                    print("✗ ERROR: Failed to open /teleop/lh_current_pose:o")
                    return False
                print("  ✓ /teleop/lh_current_pose:o created")

            self.ports["robot_state"] = yarp.Port()
            if not self.ports["robot_state"].open("/teleop/robot_state:o"):
                print("✗ ERROR: Failed to open /teleop/robot_state:o")
                return False
            print("  ✓ /teleop/robot_state:o created")

            if self.use_tactile:
                self.ports["r_hand_touch"] = yarp.BufferedPortVector()
                if not self.ports["r_hand_touch"].open("/teleop/r_hand_touch:i"):
                    print("✗ ERROR: Failed to open /teleop/r_hand_touch:i")
                    return False
                print("  ✓ /teleop/r_hand_touch:i created")

                if self.tactile_include_palm:
                    self.ports["r_hand_palm_touch"] = yarp.BufferedPortVector()
                    if not self.ports["r_hand_palm_touch"].open("/teleop/r_hand_palm_touch:i"):
                        print("✗ ERROR: Failed to open /teleop/r_hand_palm_touch:i")
                        return False
                    print("  ✓ /teleop/r_hand_palm_touch:i created")

                self.ports["touch_state"] = yarp.Port()
                if not self.ports["touch_state"].open("/teleop/touch:o"):
                    print("✗ ERROR: Failed to open /teleop/touch:o")
                    return False
                print("  ✓ /teleop/touch:o created")

            if self.use_gaze:
                self.ports["gaze_target"] = yarp.BufferedPortBottle()
                if not self.ports["gaze_target"].open("/teleop/gaze_target:i"):
                    print("✗ ERROR: Failed to open /teleop/gaze_target:i")
                    return False
                print("  ✓ /teleop/gaze_target:i created")

            # View port for recorder
            self.ports["view_out"] = yarp.BufferedPortImageRgb()
            if not self.ports["view_out"].open("/teleop/view:o"):
                print("✗ ERROR: Failed to open /teleop/view:o")
                return False
            print("  ✓ /teleop/view:o created")

            # --- Estado Interno ---
            print("[8/9] Initializing internal state...")
            self.rh_stopped = False
            self.lh_stopped = False
            self.last_touch = np.zeros(60, dtype=np.float64)
            self.last_touch_fingers = np.zeros(60, dtype=np.float64)
            self.last_touch_palm = np.zeros(0, dtype=np.float64)
            self._touch_threshold = 100.0 / 12.0
            self._touch_prev_active = np.zeros((5,), dtype=bool)
            self._touch_names = ["index", "middle", "ring", "little", "thumb"]
            self._touch_prev_palm_active = False
            self.debug_stream = os.getenv("TELEOP_DEBUG_STREAM", "0") == "1"
            self._dbg_rh_count = 0
            self._dbg_lh_count = 0
            
            self.hand_vals = {
                "open": np.array(hand_open_values).ravel(),
                "close": np.array(hand_closed_values).ravel()
            }
            print("✓ Internal state initialized")

            if self.use_tactile and self.tactile_auto_connect:
                try:
                    ok = yarp.Network.connect(self.tactile_source_port, "/teleop/r_hand_touch:i", "fast_tcp")
                    if ok:
                        print(f"  ✓ Tactile connected: {self.tactile_source_port} -> /teleop/r_hand_touch:i")
                    else:
                        print(f"  ⏳ Tactile source not connected now: {self.tactile_source_port}")

                    if self.tactile_include_palm:
                        ok_palm = yarp.Network.connect(
                            self.tactile_palm_source_port,
                            "/teleop/r_hand_palm_touch:i",
                            "fast_tcp"
                        )
                        if ok_palm:
                            print(
                                f"  ✓ Palm tactile connected: {self.tactile_palm_source_port} "
                                f"-> /teleop/r_hand_palm_touch:i"
                            )
                        else:
                            print(f"  ⏳ Palm tactile source not connected now: {self.tactile_palm_source_port}")
                except Exception as e:
                    print(f"  ✗ Tactile autoconnect error: {e}")

            # --- VR Publisher ---
            print("[9/9] Initializing VR publisher...")
            if ZMQCompressedImageTransmitter:
                try:
                    self.vr_pub = ZMQCompressedImageTransmitter(host="*", port=10505)
                    print("✓ VR Image Publisher started on port 10505")
                except Exception as e:
                    print(f"✗ Failed to start VR Publisher: {e}")
            else:
                print("✗ Warning: ZMQCompressedImageTransmitter not available")
            
            print("="*60)
            print("✓ MODULE CONFIGURATION COMPLETED SUCCESSFULLY")
            print(f"Current State: {self.state}")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"✗ ERROR during configuration: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def respond(self, command, reply):
        """
        RPC Command Handler - State Machine Control.
        
        This method is called when a command is received on the RPC port.
        It parses the command and sets the appropriate state.
        """
        cmd = command.get(0).asString()
        print(f"[RPC] Command received: '{cmd}'")
        
        if cmd == "quit":
            self.state = 'quit'
            reply.addString("Quitting module.")
            
        elif cmd == "go_home":
            print("[State] Transitioning to: going_home")
            for arm in self.cartesian_arms:
                self.yarp_interface.stop_arm_control(arm)
            if self.use_gaze:
                self.yarp_interface.stop_gaze_controller()
                self.yarp_interface.set_eyes_positions(0.0, 0.0, 0.0)
            self.yarp_interface.go_to_home_pose(wait=False)
            self.state = 'going_home'
            reply.addString("Going to home pose.")
            
        elif cmd == "start_teleop":
            print("[State] Transitioning to: teleop_active")
            self.yarp_interface.set_all_parts_to_position_direct()
            self.state = 'teleop_active'
            reply.addString("Teleop active. Processing continuous targets.")
            
        elif cmd == "stop_teleop":
            print("[State] Transitioning to: idle")
            self.state = 'idle'
            reply.addString("Teleop stopped. Idle.")
            
        elif cmd == "start_gaze":
            if self.use_gaze:
                if self.yarp_interface.igaze is None:
                    print("[CMD] Reactivating Gaze Controller...")
                    self.yarp_interface.init_gaze_controller(bind_eyes=False)
                reply.addString("Gaze controller started.")
            else:
                reply.addString("Gaze control disabled in config.")
                
        elif cmd == "stop_gaze":
            if self.use_gaze:
                self.yarp_interface.stop_gaze_controller()
                reply.addString("Gaze controller stopped.")
            else:
                reply.addString("Gaze control disabled in config.")
                
        elif cmd == "open_right":
            self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
            self.state = 'opening_hand_right'
            reply.addString("Opening right hand.")
            
        elif cmd == "close_right":
            self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
            self.state = 'closing_hand_right'
            reply.addString("Closing right hand.")
            
        elif cmd == "open_left":
            self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
            self.state = 'opening_hand_left'
            reply.addString("Opening left hand.")
            
        elif cmd == "close_left":
            self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
            self.state = 'closing_hand_left'
            reply.addString("Closing left hand.")
            
        elif cmd == "reset_scenario":
            print("[CMD] Resetting scenario...")
            script_path = str(_PROJECT_ROOT / "gazebo" / "scripts" / "reset_objects.sh")
            if os.path.exists(script_path):
                subprocess.run([script_path], check=False)
                reply.addString("Scenario reset.")
            else:
                reply.addString("Reset script not found.")
                
        elif cmd == "look_at":
            if command.size() >= 4:
                x = command.get(1).asFloat64()
                y = command.get(2).asFloat64()
                z = command.get(3).asFloat64()
                # Execute immediately if in teleop_active, otherwise store
                if self.state == 'teleop_active' and self.use_gaze:
                    if self.yarp_interface.igaze:
                        self.yarp_interface.look_at(x, y, z)
                    reply.addString("ok")
                else:
                    self.pending_gaze_target = (x, y, z)
                    self.state = 'looking_at'
                    reply.addString(f"Looking at ({x:.2f}, {y:.2f}, {z:.2f}).")
            else:
                reply.addString("Usage: look_at x y z")
        
        elif cmd == "move_right_arm":
            if command.size() >= 8:
                pos = np.array([command.get(i).asFloat64() for i in range(1, 4)])
                quat = np.array([command.get(i).asFloat64() for i in range(4, 8)])
                if "right_arm" in self.cartesian_arms:
                    self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
                    self.pending_target_rh = (pos, quat)
                    self.state = 'moving_right_arm'
                    reply.addString(f"Moving right arm to {pos}.")
                else:
                    reply.addString("error: right Cartesian control disabled (check cart_ctrl/control_arms)")
            else:
                reply.addString("Usage: move_right_arm x y z qw qx qy qz")
                
        elif cmd == "move_left_arm":
            if command.size() >= 8:
                pos = np.array([command.get(i).asFloat64() for i in range(1, 4)])
                quat = np.array([command.get(i).asFloat64() for i in range(4, 8)])
                if "left_arm" in self.cartesian_arms:
                    self.next_state_after_action = 'teleop_active' if self.state == 'teleop_active' else 'idle'
                    self.pending_target_lh = (pos, quat)
                    self.state = 'moving_left_arm'
                    reply.addString(f"Moving left arm to {pos}.")
                else:
                    reply.addString("error: left Cartesian control disabled (check cart_ctrl/control_arms)")
            else:
                reply.addString("Usage: move_left_arm x y z qw qx qy qz")
                
        else:
            reply.addString(f"Command '{cmd}' not recognized.")
            
        return True

    def getPeriod(self):
        return self.period

    def updateModule(self):
        """
        Main Update Loop - State Machine Execution.
        
        Executes actions based on current state and transitions to idle when done.
        """
        
        # === STATE: QUIT ===
        if self.state == 'quit':
            self.close()
            return False
        
        # === STATE: GOING_HOME ===
        elif self.state == 'going_home':
            if self.yarp_interface.check_motion_done():
                print("[State] Home reached. Transitioning to: idle")
                self.yarp_interface.set_all_parts_to_position_direct()
                self.state = 'idle'
            else:
                # Still moving, just update cameras and state
                self._check_cameras()
                self._send_robot_state()
                return True
        
        # === STATE: TELEOP_ACTIVE ===
        # Continuous arm updates come from Buffered ports /teleop/rh_target:i and /teleop/lh_target:i
        elif self.state == 'teleop_active':
            if "right_arm" in self.cartesian_arms and "rh_target" in self.ports:
                rh_target = self.ports["rh_target"].read(False)
                if rh_target is not None and rh_target.size() >= 7:
                    pos = np.array([rh_target.get(i).asFloat64() for i in range(3)])
                    quat = np.array([rh_target.get(i).asFloat64() for i in range(3, 7)])
                    if np.isfinite(pos).all() and np.isfinite(quat).all():
                        ok = self.yarp_interface.go_to_pose_async("right_arm", pos, quat)
                        if self.debug_stream:
                            self._dbg_rh_count += 1
                            if (self._dbg_rh_count % 20) == 0 or not ok:
                                print(f"[DBG][RH] recv={self._dbg_rh_count} ok={ok} pos={np.round(pos, 4)}")

            if "left_arm" in self.cartesian_arms and "lh_target" in self.ports:
                lh_target = self.ports["lh_target"].read(False)
                if lh_target is not None and lh_target.size() >= 7:
                    pos = np.array([lh_target.get(i).asFloat64() for i in range(3)])
                    quat = np.array([lh_target.get(i).asFloat64() for i in range(3, 7)])
                    if np.isfinite(pos).all() and np.isfinite(quat).all():
                        ok = self.yarp_interface.go_to_pose_async("left_arm", pos, quat)
                        if self.debug_stream:
                            self._dbg_lh_count += 1
                            if (self._dbg_lh_count % 20) == 0 or not ok:
                                print(f"[DBG][LH] recv={self._dbg_lh_count} ok={ok} pos={np.round(pos, 4)}")
        
        # === STATE: MOVING_RIGHT_ARM ===
        elif self.state == 'moving_right_arm':
            if self.pending_target_rh and "right_arm" in self.cartesian_arms:
                pos, quat = self.pending_target_rh
                self.yarp_interface.go_to_pose_async("right_arm", pos, quat)
                self.pending_target_rh = None
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: MOVING_LEFT_ARM ===
        elif self.state == 'moving_left_arm':
            if self.pending_target_lh and "left_arm" in self.cartesian_arms:
                pos, quat = self.pending_target_lh
                self.yarp_interface.go_to_pose_async("left_arm", pos, quat)
                self.pending_target_lh = None
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: OPENING_HAND_RIGHT ===
        elif self.state == 'opening_hand_right':
            if "right_arm" in self.actuators_to_use:
                self.rh_stopped = False
                self.yarp_interface.send_hand_positions("right_arm", self.hand_vals["open"])
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: CLOSING_HAND_RIGHT ===
        elif self.state == 'closing_hand_right':
            if "right_arm" in self.actuators_to_use:
                self.rh_stopped = False
                self.yarp_interface.send_hand_positions("right_arm", self.hand_vals["close"])
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: OPENING_HAND_LEFT ===
        elif self.state == 'opening_hand_left':
            if "left_arm" in self.actuators_to_use:
                self.lh_stopped = False
                self.yarp_interface.send_hand_positions("left_arm", self.hand_vals["open"])
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: CLOSING_HAND_LEFT ===
        elif self.state == 'closing_hand_left':
            if "left_arm" in self.actuators_to_use:
                self.lh_stopped = False
                self.yarp_interface.send_hand_positions("left_arm", self.hand_vals["close"])
            self.state = self.next_state_after_action
            self.next_state_after_action = 'idle'
        
        # === STATE: LOOKING_AT ===
        elif self.state == 'looking_at':
            if self.pending_gaze_target and self.use_gaze:
                x, y, z = self.pending_gaze_target
                if self.yarp_interface.igaze:
                    self.yarp_interface.look_at(x, y, z)
                self.pending_gaze_target = None
            self.state = 'idle'
        
        # === STATE: IDLE (Default) ===
        elif self.state == 'idle':
            pass  # Do nothing, wait for commands
        
        # === COMMON OPERATIONS (Every cycle) ===
        self._check_cameras()
        self._send_robot_state()
        
        return True

    def _check_cameras(self):
        """Update camera images and display/stream."""
        if not self.show_cameras:
            return
            
        l, r = self.yarp_interface.get_camera_images()
        
        if l is not None:
            self.l_img = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
        if r is not None:
            self.r_img = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)

        # Select monocular image (priority: right)
        img_view = None
        if self.r_img is not None:
            img_view = self.r_img.copy()
        elif self.l_img is not None:
            img_view = self.l_img.copy()

        if img_view is not None:
            # Optimization: No scaling/cropping to reduce latency and motion blur
            # The VR headset will handle the scaling
            pass

            window_name = "iCub_VR_View"
            cv2.imshow(window_name, img_view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.state = 'quit'

            # Send to VR
            if hasattr(self, 'vr_pub'):
                try:
                    # img_rgb = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB) # REMOVED: ZMQ transmitter uses OpenCV encoding which expects BGR
                    self.vr_pub.send_image(img_view)
                except Exception as e:
                    pass

            # Send to YARP (Recorder)
            if "view_out" in self.ports:
                h_i, w_i = img_view.shape[:2]
                y_img = self.ports["view_out"].prepare()
                y_img.resize(w_i, h_i)
                img_rgb_yarp = cv2.cvtColor(img_view, cv2.COLOR_BGR2RGB)
                temp_wrapper = yarp.ImageRgb()
                temp_wrapper.setExternal(img_rgb_yarp.data, w_i, h_i)
                y_img.copy(temp_wrapper)
                self.ports["view_out"].write()

    def _send_robot_state(self):
        """Publish current joint state."""
        s, _ = self.yarp_interface.get_joint_state()
        bot = yarp.Bottle()
        for v in s:
            bot.addFloat64(v)
        self.ports["robot_state"].write(bot)

        # Publish executed Cartesian pose feedback as: x y z qw qx qy qz
        if "right_arm" in self.cartesian_arms and "rh_current_pose" in self.ports:
            curr_pos, curr_quat_wxyz = self.yarp_interface._get_current_pose("right_arm")
            if curr_pos is not None and curr_quat_wxyz is not None:
                pbot = self.ports["rh_current_pose"].prepare()
                pbot.clear()
                pbot.addFloat64(float(curr_pos[0]))
                pbot.addFloat64(float(curr_pos[1]))
                pbot.addFloat64(float(curr_pos[2]))
                pbot.addFloat64(float(curr_quat_wxyz[0]))
                pbot.addFloat64(float(curr_quat_wxyz[1]))
                pbot.addFloat64(float(curr_quat_wxyz[2]))
                pbot.addFloat64(float(curr_quat_wxyz[3]))
                self.ports["rh_current_pose"].write()

        if "left_arm" in self.cartesian_arms and "lh_current_pose" in self.ports:
            curr_pos, curr_quat_wxyz = self.yarp_interface._get_current_pose("left_arm")
            if curr_pos is not None and curr_quat_wxyz is not None:
                pbot = self.ports["lh_current_pose"].prepare()
                pbot.clear()
                pbot.addFloat64(float(curr_pos[0]))
                pbot.addFloat64(float(curr_pos[1]))
                pbot.addFloat64(float(curr_pos[2]))
                pbot.addFloat64(float(curr_quat_wxyz[0]))
                pbot.addFloat64(float(curr_quat_wxyz[1]))
                pbot.addFloat64(float(curr_quat_wxyz[2]))
                pbot.addFloat64(float(curr_quat_wxyz[3]))
                self.ports["lh_current_pose"].write()

        if self.use_tactile and "r_hand_touch" in self.ports and "touch_state" in self.ports:
            touch_vec = self.ports["r_hand_touch"].read(False)
            if touch_vec is not None:
                n = touch_vec.size()
                self.last_touch_fingers = np.array([touch_vec.get(i) for i in range(n)], dtype=np.float64)
                self._debug_touch_fingers(self.last_touch_fingers)

            if self.tactile_include_palm and "r_hand_palm_touch" in self.ports:
                palm_vec = self.ports["r_hand_palm_touch"].read(False)
                if palm_vec is not None:
                    n_p = palm_vec.size()
                    self.last_touch_palm = np.array([palm_vec.get(i) for i in range(n_p)], dtype=np.float64)
                    self._debug_touch_palm(self.last_touch_palm)

            if self.tactile_include_palm and self.last_touch_palm.size > 0:
                self.last_touch = np.concatenate((self.last_touch_fingers, self.last_touch_palm))
            else:
                self.last_touch = self.last_touch_fingers

            tbot = yarp.Bottle()
            for v in self.last_touch:
                tbot.addFloat64(float(v))
            self.ports["touch_state"].write(tbot)

    def _debug_touch_fingers(self, touch_values):
        """Print debug when tactile contact appears on a finger."""
        if touch_values is None:
            return

        touch_np = np.asarray(touch_values, dtype=np.float32)
        if touch_np.size < 12:
            return

        n_fingers = min(5, touch_np.size // 12)
        means = np.array([
            float(np.mean(touch_np[i * 12:(i + 1) * 12]))
            for i in range(n_fingers)
        ], dtype=np.float32)
        active = means >= self._touch_threshold

        if self._touch_prev_active.shape[0] != n_fingers:
            self._touch_prev_active = np.zeros((n_fingers,), dtype=bool)

        newly_active = np.where(np.logical_and(active, np.logical_not(self._touch_prev_active)))[0]
        for idx in newly_active:
            name = self._touch_names[idx] if idx < len(self._touch_names) else f"finger_{idx}"
            taxel_start = idx * 12
            taxel_end = taxel_start + 11
            print(
                f"[TOUCH][DEBUG] Dedo detectado: idx={idx} ({name}) "
                f"taxels={taxel_start}-{taxel_end} mean={means[idx]:.2f}"
            )

        self._touch_prev_active = active

    def _debug_touch_palm(self, palm_values):
        """Print debug when tactile contact appears on the palm."""
        if palm_values is None:
            return

        palm_np = np.asarray(palm_values, dtype=np.float32)
        if palm_np.size == 0:
            return

        mean_val = float(np.mean(palm_np))
        active = mean_val >= self._touch_threshold

        if active and not self._touch_prev_palm_active:
            print(
                f"[TOUCH][DEBUG] Palma detectada: taxels=0-{palm_np.size - 1} "
                f"mean={mean_val:.2f}"
            )

        self._touch_prev_palm_active = active

    def interruptModule(self):
        print("\n[Interrupt] Stopping ports...")
        if hasattr(self, 'ports'):
            for p in self.ports.values():
                p.interrupt()
        if hasattr(self, 'cmd_port'):
            self.cmd_port.interrupt()
        return True

    def close(self):
        if getattr(self, '_closed', False):
            return True
        self._closed = True

        print("\n[Cleanup] Releasing YARP and OpenCV resources...")
        
        if hasattr(self, 'yarp_interface'):
            print("  [Exit] Sending robot to Home before exit...")
            try:
                if hasattr(self, 'cartesian_arms'):
                    for arm in self.cartesian_arms:
                        self.yarp_interface.stop_arm_control(arm)
                if hasattr(self, 'use_gaze') and self.use_gaze:
                    self.yarp_interface.stop_gaze_controller()
                self.yarp_interface.go_to_home_pose(wait=True)
            except Exception as e:
                print(f"  Warning: Could not go home on exit: {e}")

        if hasattr(self, 'cmd_port'):
            self.cmd_port.close()
            print("  ✓ Command port closed")
            
        if hasattr(self, 'ports'):
            for name, port in self.ports.items():
                try:
                    port.close()
                    print(f"  ✓ Port {name} closed")
                except:
                    pass
        
        if hasattr(self, 'yarp_interface'):
            try:
                self.yarp_interface.close()
                print("  ✓ YARP interface closed")
            except:
                pass

        cv2.destroyAllWindows()
        return True


if __name__ == '__main__':
    module = None
    try:
        print("\n" + "="*60)
        print("STARTING TELEOP MODULE (STATE MACHINE)")
        print("="*60 + "\n")
        
        yarp.Network.init()
        if not yarp.Network.checkNetwork():
            print("✗ ERROR: YARP network is not available!")
            print("  Please start yarpserver first with: yarpserver --write")
            sys.exit(1)
        print("✓ YARP network is available\n")
        
        module = TeleopModuleSM()
        rf = yarp.ResourceFinder()
        rf.configure(sys.argv)
        
        print("Starting module main loop...")
        print("\nAvailable RPC commands (use: yarp rpc /teleop/rpc:i):")
        print("  go_home         - Move robot to home pose")
        print("  start_teleop    - Enable continuous target processing")
        print("  stop_teleop     - Stop teleop, return to idle") 
        print("  open_right/left - Open hand")
        print("  close_right/left- Close hand")
        print("  look_at x y z   - Look at 3D point")
        print("  start_gaze      - Enable gaze controller")
        print("  (Touch) Input   - /teleop/r_hand_touch:i")
        print("  (Touch) Input   - /teleop/r_hand_palm_touch:i")
        print("  (Touch) Output  - /teleop/touch:o")
        print("  reset_scenario  - Reset objects")
        print("  quit            - Close module\n")
        
        ret = module.runModule(rf)
        
        if ret:
            print("\n✓ Module exited successfully")
        else:
            print("\n✗ Module exited with error")
            
    except KeyboardInterrupt:
        print("\n\n✓ Module stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'module' in locals() and module is not None:
            module.close()
        yarp.Network.fini()
        print("✓ Cleanup completed\n")
        sys.exit(0)
