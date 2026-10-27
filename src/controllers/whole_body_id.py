#!/usr/bin/env python3
"""
Whole-Body Task-Space Inverse Dynamics (TSID) & Cartesian Impedance Controller for iCub
using Bipedal Locomotion Framework (BLF).

Manages:
  - Bimanual end-effector tracking / compliance in SE(3) (left & right hands)
  - Head gaze orientation tracking in SO(3)
  - Joint posture regularization in null-space
  - Joint torque L2 regularization
  - Rigid-body dynamics and fixed-base constraints via blf.tsid.QPFixedBaseTSID
  - Physical per-joint torque saturation limits

Compatible with simulated models (iCubGazeboV2_5, iCubGazeboV2_5_fixed, iCubGazeboV2_7,
iCubGazeboV2_7_fixed, iCubGazeboV2_5_visuomanip), MuJoCo (icub_visuomanip_idyn_limits.xml),
and the real pedestal-mounted iCub V2.7.
"""

import xml.etree.ElementTree as ET
import numpy as np
from datetime import timedelta
from typing import Tuple, Optional, List, Dict

import bipedal_locomotion_framework.bindings as blf
import manifpy as manif
import idyntree.swig as idyn
import icub_models

from .whole_body_ik import rotmat_to_quat, rotmat_to_quat_wxyz, quat_wxyz_to_rotmat


def resolve_urdf_model_file(robot_model_name: str) -> str:
    """
    Resolve the URDF file path for an iCub model name in icub_models.
    Automatically maps SDF-only fixed-base wrapper models (e.g. 'iCubGazeboV2_5_fixed',
    'iCubGazeboV2_7_fixed', '*_feet_fixed') to their underlying URDF model.
    """
    try:
        return str(icub_models.get_model_file(robot_model_name))
    except FileNotFoundError:
        for suffix in ("_feet_fixed", "_fixed"):
            if robot_model_name.endswith(suffix):
                base_model = robot_model_name[: -len(suffix)]
                return str(icub_models.get_model_file(base_model))
        raise


def load_urdf_joint_damping(model_file: str, joints_list: List[str]) -> np.ndarray:
    """
    Extract per-joint viscous damping coefficients [N*m*s/rad] from <dynamics damping="..."/>
    in the URDF model file.
    """
    damping = np.zeros(len(joints_list), dtype=np.float64)
    try:
        tree = ET.parse(model_file)
        root = tree.getroot()
        joint_map = {}
        for j in root.findall("joint"):
            jname = j.attrib.get("name")
            dyn = j.find("dynamics")
            if jname and dyn is not None and "damping" in dyn.attrib:
                joint_map[jname] = float(dyn.attrib["damping"])
        for idx, name in enumerate(joints_list):
            if name in joint_map:
                damping[idx] = joint_map[name]
    except Exception:
        pass
    return damping


class iCubWholeBodyID:
    """
    Whole-Body Task-Space Inverse Dynamics (TSID) Controller for fixed-base iCub.
    Computes optimal joint torques tau* in R^20 [N*m] solving a constrained QP.
    Supports switching between stiff trajectory tracking and compliant Cartesian impedance.
    """

    # 20 Upper-Body Degrees of Freedom
    JOINTS_LIST = [
        "torso_yaw", "torso_roll", "torso_pitch",
        "neck_pitch", "neck_roll", "neck_yaw",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
        "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw",
    ]

    DEFAULT_HOME_POSTURE = np.array([
        # Torso: yaw, roll, pitch
        0.0, 0.0, 0.17453292519943295,
        # Neck: pitch, roll, yaw
        0.2617993877991494, 0.0, 0.0,
        # Left arm: pitch, roll, yaw, elbow, wrist_prosup, wrist_pitch, wrist_yaw
        -1.0471975511965976, 0.3490658503988659, 0.0, 1.2217304763960306, 0.0, -0.17453292519943295, 0.0,
        # Right arm: pitch, roll, yaw, elbow, wrist_prosup, wrist_pitch, wrist_yaw
        -1.0471975511965976, 0.3490658503988659, 0.0, 1.2217304763960306, 0.0, -0.17453292519943295, 0.0,
    ], dtype=np.float64)

    # Nominal physical torque limits [N*m] for iCub upper body
    DEFAULT_MAX_TORQUES = np.array([
        # Torso (yaw, roll, pitch)
        40.0, 40.0, 40.0,
        # Neck (pitch, roll, yaw)
        6.0, 6.0, 6.0,
        # Left arm (3 shoulder, 1 elbow, 3 wrist)
        25.0, 25.0, 20.0, 18.0, 6.0, 6.0, 6.0,
        # Right arm (3 shoulder, 1 elbow, 3 wrist)
        25.0, 25.0, 20.0, 18.0, 6.0, 6.0, 6.0,
    ], dtype=np.float64)

    LEFT_HAND_FRAME = "l_hand_dh_frame"
    RIGHT_HAND_FRAME = "r_hand_dh_frame"
    HEAD_FRAME = "head"
    ROBOT_ACC_VAR = "robotAcceleration"
    JOINT_TORQUES_VAR = "jointTorques"

    def __init__(
        self,
        sampling_time: float = 0.01,
        robot_model_name: str = "iCubGazeboV2_5",
        kp_cartesian: float = 100.0,
        kd_cartesian: Optional[float] = None,
        kp_angular: float = 80.0,
        kd_angular: Optional[float] = None,
        kp_posture: float = 25.0,
        kd_posture: Optional[float] = None,
        weight_hands: float = 100.0,
        weight_head: float = 20.0,
        weight_posture: Optional[float] = None,
        weight_posture_torso: float = 2.0,
        weight_posture_neck: float = 0.05,
        weight_posture_arms: float = 0.1,
        weight_torque_reg: float = 1e-5,
        impedance_mode: bool = False,
        impedance_stiffness_scale: float = 0.35,
        max_joint_torques: Optional[np.ndarray] = None,
        compensate_urdf_damping: Optional[bool] = None,
        joint_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        joint_limit_horizon_s: float = 0.1,
    ):
        self.sampling_time = sampling_time
        self.dT = timedelta(seconds=sampling_time)
        self.robot_model_name = robot_model_name
        self.nr_dofs = len(self.JOINTS_LIST)

        # Store nominal (stiff tracking) gains
        self.nominal_kp_cartesian = float(kp_cartesian)
        self.nominal_kd_cartesian = (
            float(kd_cartesian) if kd_cartesian is not None else 2.0 * np.sqrt(self.nominal_kp_cartesian)
        )
        self.nominal_kp_angular = float(kp_angular)
        self.nominal_kd_angular = (
            float(kd_angular) if kd_angular is not None else 2.0 * np.sqrt(self.nominal_kp_angular)
        )
        self.nominal_kp_posture = float(kp_posture)
        self.nominal_kd_posture = (
            float(kd_posture) if kd_posture is not None else 2.0 * np.sqrt(self.nominal_kp_posture)
        )

        self.weight_hands = float(weight_hands)
        self.weight_head = float(weight_head)
        self.weight_posture = float(weight_posture) if weight_posture is not None else None
        self.weight_posture_torso = float(weight_posture_torso)
        self.weight_posture_neck = float(weight_posture_neck)
        self.weight_posture_arms = float(weight_posture_arms)
        self.weight_torque_reg = float(weight_torque_reg)

        self.impedance_mode = bool(impedance_mode)
        self.impedance_stiffness_scale = float(impedance_stiffness_scale)

        if max_joint_torques is not None:
            self.max_joint_torques = np.asarray(max_joint_torques, dtype=np.float64).copy()
            if self.max_joint_torques.shape != (self.nr_dofs,):
                raise ValueError(f"max_joint_torques must have shape ({self.nr_dofs},)")
        else:
            self.max_joint_torques = self.DEFAULT_MAX_TORQUES.copy()

        # 1. Load reduced URDF model
        self.model_loader = idyn.ModelLoader()
        model_file = resolve_urdf_model_file(robot_model_name)
        if not self.model_loader.loadReducedModelFromFile(model_file, self.JOINTS_LIST):
            raise RuntimeError(f"Failed to load reduced model from {model_file}")

        # Joint limits {name: (min_rad, max_rad)} of config joint_limits_deg (controllers/joint_limits.py)
        # replace the URDF ones; the joint limits task below enforces them.
        model = self.model_loader.model()
        for j_name, (lo, hi) in (joint_limits or {}).items():
            model.getJoint(model.getJointIndex(j_name)).setPosLimits(0, float(lo), float(hi))
        self.q_min = np.array([model.getJoint(model.getJointIndex(j)).getMinPosLimit(0) for j in self.JOINTS_LIST])
        self.q_max = np.array([model.getJoint(model.getJointIndex(j)).getMaxPosLimit(0) for j in self.JOINTS_LIST])
        self.joint_limit_horizon_s = float(joint_limit_horizon_s)

        # Load URDF viscous joint damping coefficients (Gazebo models and icub_visuomanip_idyn_limits.xml define damping=1.0 Nm*s/rad)
        self.urdf_joint_damping = load_urdf_joint_damping(model_file, self.JOINTS_LIST)
        if compensate_urdf_damping is None:
            self.compensate_urdf_damping = "gazebo" in robot_model_name.lower()
        else:
            self.compensate_urdf_damping = bool(compensate_urdf_damping)

        self.kindyn = idyn.KinDynComputations()
        if not self.kindyn.loadRobotModel(model):
            raise RuntimeError("Failed to load robot model into KinDynComputations")

        if not self.kindyn.setFrameVelocityRepresentation(idyn.MIXED_REPRESENTATION):
            raise RuntimeError("Failed to set MIXED_REPRESENTATION on KinDynComputations")

        # Initialize state buffers to safe home posture
        self.s = self.DEFAULT_HOME_POSTURE.copy()
        self.s_dot = np.zeros(self.nr_dofs, dtype=np.float64)
        self.base_pose = np.eye(4, dtype=np.float64)
        self.base_vel = np.zeros(6, dtype=np.float64)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)

        # 2. Configure QPFixedBaseTSID Solver
        self.qp_tsid = blf.tsid.QPFixedBaseTSID()
        qp_param_handler = blf.parameters_handler.StdParametersHandler()
        qp_param_handler.set_parameter_string("robot_acceleration_variable_name", self.ROBOT_ACC_VAR)
        qp_param_handler.set_parameter_string("joint_torques_variable_name", self.JOINT_TORQUES_VAR)
        qp_param_handler.set_parameter_int("max_number_of_contacts", 0)
        qp_param_handler.set_parameter_bool("verbosity", False)

        if not self.qp_tsid.set_kin_dyn(self.kindyn):
            raise RuntimeError("Failed to set kinDyn in QPFixedBaseTSID")
        if not self.qp_tsid.initialize(handler=qp_param_handler):
            raise RuntimeError("Failed to initialize QPFixedBaseTSID")

        # Variables handler (QPFixedBaseTSID automatically adds 'base_contact_wrench' during finalize)
        self.var_handler = blf.system.VariablesHandler()
        self.var_handler.add_variable(self.ROBOT_ACC_VAR, self.nr_dofs + 6)
        self.var_handler.add_variable(self.JOINT_TORQUES_VAR, self.nr_dofs)

        # Compute effective initial gains based on impedance_mode
        kp_lin, kd_lin, kp_ang, kd_ang, kp_post, kd_post = self._compute_active_gains()

        # 3. Create Left Hand Task (SE3)
        self.l_hand_task = blf.tsid.SE3Task()
        self.l_hand_task.set_kin_dyn(self.kindyn)
        self._init_se3_task(self.l_hand_task, self.LEFT_HAND_FRAME, kp_lin, kd_lin, kp_ang, kd_ang)
        if not self.qp_tsid.add_task(
            self.l_hand_task, "left_hand_task", 1, [self.weight_hands] * 6
        ):
            raise RuntimeError("Failed to add left_hand_task to QPFixedBaseTSID")

        # 4. Create Right Hand Task (SE3)
        self.r_hand_task = blf.tsid.SE3Task()
        self.r_hand_task.set_kin_dyn(self.kindyn)
        self._init_se3_task(self.r_hand_task, self.RIGHT_HAND_FRAME, kp_lin, kd_lin, kp_ang, kd_ang)
        if not self.qp_tsid.add_task(
            self.r_hand_task, "right_hand_task", 1, [self.weight_hands] * 6
        ):
            raise RuntimeError("Failed to add right_hand_task to QPFixedBaseTSID")

        # 5. Create Head Gaze Task (SO3)
        self.head_task = blf.tsid.SO3Task()
        self.head_task.set_kin_dyn(self.kindyn)
        self._init_so3_task(self.head_task, self.HEAD_FRAME, kp_ang, kd_ang)
        if not self.qp_tsid.add_task(
            self.head_task, "head_gaze_task", 1, [self.weight_head] * 3
        ):
            raise RuntimeError("Failed to add head_gaze_task to QPFixedBaseTSID")

        # 6. Create Joint Posture Tracking Task (Null-space regularization)
        self.posture_task = blf.tsid.JointTrackingTask()
        self.posture_task.set_kin_dyn(self.kindyn)
        self._init_posture_task(self.posture_task, kp_post, kd_post)
        if weight_posture is not None:
            posture_weights = [float(weight_posture)] * self.nr_dofs
        else:
            posture_weights = (
                [float(weight_posture_torso)] * 3
                + [float(weight_posture_neck)] * 3
                + [float(weight_posture_arms)] * 14
            )

        if not self.qp_tsid.add_task(
            self.posture_task, "posture_task", 1, posture_weights
        ):
            raise RuntimeError("Failed to add posture_task to QPFixedBaseTSID")

        # 7. Create Joint Torque Regularization Task (L2 penalty on jointTorques)
        reg_handler = blf.parameters_handler.StdParametersHandler()
        reg_handler.set_parameter_string("variable_name", self.JOINT_TORQUES_VAR)
        reg_handler.set_parameter_int("variable_size", self.nr_dofs)
        self.torque_reg_task = blf.tsid.VariableRegularizationTask()
        if not self.torque_reg_task.initialize(reg_handler):
            raise RuntimeError("Failed to initialize VariableRegularizationTask for jointTorques")
        if not self.qp_tsid.add_task(
            self.torque_reg_task, "torque_reg_task", 1, [self.weight_torque_reg] * self.nr_dofs
        ):
            raise RuntimeError("Failed to add torque_reg_task to QPFixedBaseTSID")

        # 8. Joint limits (hard): TSID has no joint limits task, so the joint accelerations are
        # bounded to those that stop the joint at its limit within joint_limit_horizon_s T:
        #   q + dq*T + ddq*T^2/2 <= q_max  ->  ddq <= 2*(q_max - q - dq*T)/T^2   (same for q_min)
        lim_handler = blf.parameters_handler.StdParametersHandler()
        lim_handler.set_parameter_string("variable_name", self.ROBOT_ACC_VAR)
        lim_handler.set_parameter_int("variable_size", self.nr_dofs + 6)
        self.joint_limits_task = blf.tsid.VariableFeasibleRegionTask()
        if not self.joint_limits_task.initialize(lim_handler):
            raise RuntimeError("Failed to initialize the joint limits task (VariableFeasibleRegionTask)")
        # robotAcceleration = [base (6), joints (nr_dofs)]: the bounds act on the joints only
        self._limits_C = np.asfortranarray(np.hstack([np.zeros((self.nr_dofs, 6)), np.eye(self.nr_dofs)]))
        self._update_joint_limits_task()
        if not self.qp_tsid.add_task(self.joint_limits_task, "joint_limits_task", 0):
            raise RuntimeError("Failed to add joint_limits_task to QPFixedBaseTSID")

        # Finalize QP solver
        if not self.qp_tsid.finalize(self.var_handler):
            raise RuntimeError("Failed to finalize QPFixedBaseTSID")

        # Initialize default setpoints
        self.posture_ref = self.DEFAULT_HOME_POSTURE.copy()
        self.posture_vel_ref = np.zeros(self.nr_dofs, dtype=np.float64)
        self.posture_acc_ref = np.zeros(self.nr_dofs, dtype=np.float64)
        self.posture_task.set_set_point(self.posture_ref, self.posture_vel_ref, self.posture_acc_ref)

        self.torque_ref = np.zeros(self.nr_dofs, dtype=np.float64)
        self.torque_reg_task.set_set_point(self.torque_ref)

        # Initialize Cartesian setpoints to current initial poses
        pos_l0, rot_l0 = self.get_current_left_hand_pose()
        pos_r0, rot_r0 = self.get_current_right_hand_pose()
        rot_h0 = self.get_current_head_orientation()
        self.set_desired_left_hand(pos_l0, rot_l0)
        self.set_desired_right_hand(pos_r0, rot_r0)
        self.set_desired_head_orientation(rot_h0)

        # Cache last valid solution
        self.last_torques = np.zeros(self.nr_dofs, dtype=np.float64)
        self.last_joint_accelerations = np.zeros(self.nr_dofs, dtype=np.float64)

    def _compute_active_gains(self) -> Tuple[float, float, float, float, float, float]:
        """Compute active (kp_lin, kd_lin, kp_ang, kd_ang, kp_post, kd_post) based on mode."""
        if self.impedance_mode:
            scale = self.impedance_stiffness_scale
            kp_lin = self.nominal_kp_cartesian * scale
            kd_lin = 2.0 * np.sqrt(kp_lin)
            kp_ang = self.nominal_kp_angular * scale
            kd_ang = 2.0 * np.sqrt(kp_ang)
            kp_post = self.nominal_kp_posture * scale
            kd_post = 2.0 * np.sqrt(kp_post)
        else:
            kp_lin = self.nominal_kp_cartesian
            kd_lin = self.nominal_kd_cartesian
            kp_ang = self.nominal_kp_angular
            kd_ang = self.nominal_kd_angular
            kp_post = self.nominal_kp_posture
            kd_post = self.nominal_kd_posture
        return kp_lin, kd_lin, kp_ang, kd_ang, kp_post, kd_post

    def _init_se3_task(
        self,
        task: blf.tsid.SE3Task,
        frame_name: str,
        kp_lin: float,
        kd_lin: float,
        kp_ang: float,
        kd_ang: float,
    ):
        handler = blf.parameters_handler.StdParametersHandler()
        handler.set_parameter_string("robot_acceleration_variable_name", self.ROBOT_ACC_VAR)
        handler.set_parameter_string("frame_name", frame_name)
        handler.set_parameter_vector_bool("mask", [True, True, True])
        handler.set_parameter_float("kp_linear", float(kp_lin))
        handler.set_parameter_float("kd_linear", float(kd_lin))
        handler.set_parameter_float("kp_angular", float(kp_ang))
        handler.set_parameter_float("kd_angular", float(kd_ang))
        if not task.initialize(param_handler=handler):
            raise RuntimeError(f"Failed to initialize SE3Task for frame '{frame_name}'")

    def _init_so3_task(
        self,
        task: blf.tsid.SO3Task,
        frame_name: str,
        kp_ang: float,
        kd_ang: float,
    ):
        handler = blf.parameters_handler.StdParametersHandler()
        handler.set_parameter_string("robot_acceleration_variable_name", self.ROBOT_ACC_VAR)
        handler.set_parameter_string("frame_name", frame_name)
        handler.set_parameter_float("kp_angular", float(kp_ang))
        handler.set_parameter_float("kd_angular", float(kd_ang))
        if not task.initialize(param_handler=handler):
            raise RuntimeError(f"Failed to initialize SO3Task for frame '{frame_name}'")

    def _init_posture_task(
        self,
        task: blf.tsid.JointTrackingTask,
        kp_post: float,
        kd_post: float,
    ):
        handler = blf.parameters_handler.StdParametersHandler()
        handler.set_parameter_string("robot_acceleration_variable_name", self.ROBOT_ACC_VAR)
        handler.set_parameter_vector_float("kp", [float(kp_post)] * self.nr_dofs)
        handler.set_parameter_vector_float("kd", [float(kd_post)] * self.nr_dofs)
        if not task.initialize(param_handler=handler):
            raise RuntimeError("Failed to initialize JointTrackingTask for posture")

    def set_impedance_mode(self, enabled: bool, stiffness_scale: Optional[float] = None):
        """
        Switch dynamically between stiff trajectory tracking (enabled=False) and
        compliant Cartesian impedance control (enabled=True).
        """
        self.impedance_mode = bool(enabled)
        if stiffness_scale is not None:
            self.impedance_stiffness_scale = float(stiffness_scale)

        kp_lin, kd_lin, kp_ang, kd_ang, kp_post, kd_post = self._compute_active_gains()

        self._init_se3_task(self.l_hand_task, self.LEFT_HAND_FRAME, kp_lin, kd_lin, kp_ang, kd_ang)
        self._init_se3_task(self.r_hand_task, self.RIGHT_HAND_FRAME, kp_lin, kd_lin, kp_ang, kd_ang)
        self._init_so3_task(self.head_task, self.HEAD_FRAME, kp_ang, kd_ang)
        self._init_posture_task(self.posture_task, kp_post, kd_post)
        self.posture_task.set_set_point(self.posture_ref, self.posture_vel_ref, self.posture_acc_ref)

    def set_base_pose(self, position: np.ndarray, rotation_matrix: np.ndarray):
        """Set world-to-root_link homogeneous transform (defaults to I_4 for root_link frame)."""
        self.base_pose = np.eye(4, dtype=np.float64)
        self.base_pose[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
        self.base_pose[:3, 3] = np.asarray(position, dtype=np.float64)
        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)

    def update_robot_state(
        self,
        joint_positions: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None,
    ):
        """Update robot configuration and velocity in KinDynComputations."""
        self.s = np.asarray(joint_positions, dtype=np.float64)
        if joint_velocities is not None:
            self.s_dot = np.asarray(joint_velocities, dtype=np.float64)
        else:
            self.s_dot.fill(0.0)

        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)
        self._update_joint_limits_task()

    def _update_joint_limits_task(self) -> None:
        """Joint acceleration bounds that stop each joint at its limit within the horizon."""
        T = self.joint_limit_horizon_s
        reach = self.s + self.s_dot * T
        lower = 2.0 * (self.q_min - reach) / T**2
        upper = 2.0 * (self.q_max - reach) / T**2
        # Already beyond a limit: the bound pulls it back (lower <= upper always holds)
        lower = np.minimum(lower, upper)
        self.joint_limits_task.set_feasible_region(self._limits_C, lower.reshape(-1, 1), upper.reshape(-1, 1))

    def get_current_frame_pose(self, frame_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return position (3,) and rotation matrix (3,3) in world frame for any link/frame."""
        transform = self.kindyn.getWorldTransform(frame_name)
        mat = transform.asHomogeneousTransform().toNumPy()
        pos = mat[:3, 3].copy()
        rot = mat[:3, :3].copy()
        return pos, rot

    def get_current_left_hand_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_current_frame_pose(self.LEFT_HAND_FRAME)

    def get_current_right_hand_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_current_frame_pose(self.RIGHT_HAND_FRAME)

    def get_current_head_orientation(self) -> np.ndarray:
        _, rot = self.get_current_frame_pose(self.HEAD_FRAME)
        return rot

    def get_current_head_quaternion(self) -> np.ndarray:
        """Return current head orientation as quaternion [w, x, y, z]."""
        return rotmat_to_quat_wxyz(self.get_current_head_orientation())

    def get_gravity_compensation_torques(self) -> np.ndarray:
        """
        Return the generalized bias forces (gravity + Coriolis/centrifugal) for the 20 joints [N*m].
        """
        bias = idyn.FreeFloatingGeneralizedTorques(self.model_loader.model())
        if not self.kindyn.generalizedBiasForces(bias):
            return np.zeros(self.nr_dofs, dtype=np.float64)
        return bias.jointTorques().toNumPy().copy()

    @staticmethod
    def _to_se3_tangent(vec: Optional[np.ndarray], name: str) -> manif.SE3Tangent:
        if vec is None:
            return manif.SE3Tangent.Zero()
        arr = np.asarray(vec, dtype=np.float64).flatten()
        if len(arr) == 3:
            vec_6d = np.zeros(6, dtype=np.float64)
            vec_6d[:3] = arr
        elif len(arr) == 6:
            vec_6d = arr
        else:
            raise ValueError(f"{name} must have length 3 or 6, got {len(arr)}")
        return manif.SE3Tangent(vec_6d)

    def set_desired_left_hand(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        feedforward_velocity: Optional[np.ndarray] = None,
        feedforward_acceleration: Optional[np.ndarray] = None,
    ):
        """Set 6D SE(3) pose, velocity, and acceleration setpoints for the left hand."""
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_H_l_hand = manif.SE3(position=pos, quaternion=quat)

        tangent_vel = self._to_se3_tangent(feedforward_velocity, "feedforward_velocity")
        tangent_acc = self._to_se3_tangent(feedforward_acceleration, "feedforward_acceleration")

        self.l_hand_task.set_set_point(
            I_H_F=I_H_l_hand,
            mixed_velocity=tangent_vel,
            mixed_acceleration=tangent_acc,
        )

    def set_desired_right_hand(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        feedforward_velocity: Optional[np.ndarray] = None,
        feedforward_acceleration: Optional[np.ndarray] = None,
    ):
        """Set 6D SE(3) pose, velocity, and acceleration setpoints for the right hand."""
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_H_r_hand = manif.SE3(position=pos, quaternion=quat)

        tangent_vel = self._to_se3_tangent(feedforward_velocity, "feedforward_velocity")
        tangent_acc = self._to_se3_tangent(feedforward_acceleration, "feedforward_acceleration")

        self.r_hand_task.set_set_point(
            I_H_F=I_H_r_hand,
            mixed_velocity=tangent_vel,
            mixed_acceleration=tangent_acc,
        )

    def set_desired_head_orientation(
        self,
        rotation_matrix: np.ndarray,
        feedforward_angular_velocity: Optional[np.ndarray] = None,
        feedforward_angular_acceleration: Optional[np.ndarray] = None,
    ):
        """Set 3D SO(3) orientation, angular velocity, and angular acceleration for the head."""
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_R_head = manif.SO3(quaternion=quat)

        if feedforward_angular_velocity is not None:
            w = np.asarray(feedforward_angular_velocity, dtype=np.float64).flatten()
            tangent_ang_vel = manif.SO3Tangent(w)
        else:
            tangent_ang_vel = manif.SO3Tangent.Zero()

        if feedforward_angular_acceleration is not None:
            dw = np.asarray(feedforward_angular_acceleration, dtype=np.float64).flatten()
            tangent_ang_acc = manif.SO3Tangent(dw)
        else:
            tangent_ang_acc = manif.SO3Tangent.Zero()

        self.head_task.set_set_point(
            I_R_F=I_R_head,
            angular_velocity=tangent_ang_vel,
            angular_acceleration=tangent_ang_acc,
        )

    def set_desired_head_quaternion(
        self,
        quat_wxyz: np.ndarray,
        feedforward_angular_velocity: Optional[np.ndarray] = None,
        feedforward_angular_acceleration: Optional[np.ndarray] = None,
    ):
        """Set 3D SO(3) setpoint for head orientation from quaternion [w, x, y, z]."""
        self.set_desired_head_orientation(
            quat_wxyz_to_rotmat(quat_wxyz),
            feedforward_angular_velocity=feedforward_angular_velocity,
            feedforward_angular_acceleration=feedforward_angular_acceleration,
        )

    def set_posture_reference(
        self,
        joint_positions: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None,
        joint_accelerations: Optional[np.ndarray] = None,
    ):
        """Set null-space joint posture reference (position, velocity, acceleration)."""
        self.posture_ref = np.asarray(joint_positions, dtype=np.float64).copy()
        self.posture_vel_ref = (
            np.asarray(joint_velocities, dtype=np.float64).copy()
            if joint_velocities is not None
            else np.zeros(self.nr_dofs, dtype=np.float64)
        )
        self.posture_acc_ref = (
            np.asarray(joint_accelerations, dtype=np.float64).copy()
            if joint_accelerations is not None
            else np.zeros(self.nr_dofs, dtype=np.float64)
        )
        self.posture_task.set_set_point(
            joint_position=self.posture_ref,
            joint_velocity=self.posture_vel_ref,
            joint_acceleration=self.posture_acc_ref,
        )

    def set_torque_regularization_reference(self, torque_ref: np.ndarray):
        """Set reference torque vector (e.g. zero or gravity bias) for L2 torque regularization."""
        self.torque_ref = np.asarray(torque_ref, dtype=np.float64).copy()
        self.torque_reg_task.set_set_point(self.torque_ref)

    def step(self) -> np.ndarray:
        """
        Solve the TSID QP and return the optimal clamped joint torques (20,) in [N*m].
        """
        if not self.qp_tsid.advance():
            raise RuntimeError("QPFixedBaseTSID solver failed to advance.")

        if not self.qp_tsid.is_output_valid():
            raise RuntimeError("QPFixedBaseTSID output is marked as invalid.")

        output = self.qp_tsid.get_output()
        raw_torques = np.asarray(output.joint_torques, dtype=np.float64)
        raw_acc = np.asarray(output.joint_accelerations, dtype=np.float64)

        if np.any(np.isnan(raw_torques)) or np.any(np.isinf(raw_torques)):
            raise RuntimeError("NaN or Inf detected in TSID computed joint torques.")

        # Compensate for URDF viscous joint damping in simulation if active
        if self.compensate_urdf_damping:
            raw_torques = raw_torques + self.urdf_joint_damping * self.s_dot

        # Apply physical safety saturation limits
        clamped_torques = np.clip(raw_torques, -self.max_joint_torques, self.max_joint_torques)

        self.last_torques = clamped_torques
        self.last_joint_accelerations = raw_acc
        return clamped_torques.copy()

    def get_last_joint_accelerations(self) -> np.ndarray:
        """Return the optimal joint accelerations (20,) [rad/s^2] from the last QP solve."""
        return self.last_joint_accelerations.copy()

