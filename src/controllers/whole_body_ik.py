#!/usr/bin/env python3
"""
Whole-Body Inverse Kinematics Controller for iCub using Bipedal Locomotion Framework (BLF).
Manages bimanual end-effector tracking (SE3), head gaze tracking (SO3),
joint limits, velocity limits, and posture regularization in null space.
"""

import numpy as np
from datetime import timedelta
from typing import List, Tuple, Optional, Dict

import bipedal_locomotion_framework.bindings as blf
import manifpy as manif
import idyntree.swig as idyn
import icub_models


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w] (manifpy convention)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z] (MuJoCo / LeRobot convention)."""
    xyzw = rotmat_to_quat(R)
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    if wxyz[0] < 0:
        wxyz = -wxyz
    return wxyz


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=np.float64).flatten()
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


class iCubWholeBodyIK:
    # 20 Upper-Body Degrees of Freedom
    JOINTS_LIST = [
        "torso_yaw", "torso_roll", "torso_pitch",
        "neck_pitch", "neck_roll", "neck_yaw",
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
        "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
        "r_wrist_prosup", "r_wrist_pitch", "r_wrist_yaw"
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

    LEFT_HAND_FRAME = "l_hand_dh_frame"
    RIGHT_HAND_FRAME = "r_hand_dh_frame"
    HEAD_FRAME = "head"

    def __init__(
        self,
        sampling_time: float = 0.01,
        robot_model_name: str = "iCubGazeboV2_5",
        kp_cartesian: float = 20.0,
        kp_angular: float = 15.0,
        weight_hands: float = 180.0,
        weight_head: float = 5.0,
        weight_posture: Optional[float] = None,
        weight_posture_torso: float = 80.0,
        weight_posture_neck: float = 0.05,
        weight_posture_arms: float = 0.05,
        max_joint_velocity: float = 1.5,
        torso_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.sampling_time = sampling_time
        self.dT = timedelta(seconds=sampling_time)
        self.nr_dofs = len(self.JOINTS_LIST)

        # 1. Load kinematic model (automatically resolving *_fixed SDF wrappers to base URDF)
        self.model_loader = idyn.ModelLoader()
        try:
            model_file = str(icub_models.get_model_file(robot_model_name))
        except FileNotFoundError:
            model_file = None
            for suffix in ("_feet_fixed", "_fixed"):
                if robot_model_name.endswith(suffix):
                    base_model = robot_model_name[: -len(suffix)]
                    model_file = str(icub_models.get_model_file(base_model))
                    break
            if model_file is None:
                raise
        if not self.model_loader.loadReducedModelFromFile(model_file, self.JOINTS_LIST):
            raise RuntimeError(f"Failed to load reduced model from {model_file}")

        # Enforce custom torso limits if provided (e.g. from torso_limits_deg in config)
        model = self.model_loader.model()
        if torso_limits:
            for t_key, j_name in (("pitch", "torso_pitch"), ("roll", "torso_roll"), ("yaw", "torso_yaw")):
                if t_key in torso_limits and len(torso_limits[t_key]) == 2:
                    jidx = model.getJointIndex(j_name)
                    lo, hi = float(torso_limits[t_key][0]), float(torso_limits[t_key][1])
                    model.getJoint(jidx).setPosLimits(0, lo, hi)

        self.kindyn = idyn.KinDynComputations()
        if not self.kindyn.loadRobotModel(model):
            raise RuntimeError("Failed to load robot model into KinDynComputations")

        if not self.kindyn.setFrameVelocityRepresentation(idyn.MIXED_REPRESENTATION):
            raise RuntimeError("Failed to set frame velocity representation")

        # Initial state buffers set to safe home posture
        self.s = self.DEFAULT_HOME_POSTURE.copy()
        self.s_dot = np.zeros(self.nr_dofs)
        self.base_pose = np.eye(4)
        self.base_vel = np.zeros(6)
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)

        # 2. Configure QP Fixed Base IK Solver
        self.qp_ik = blf.ik.QPFixedBaseInverseKinematics()
        qp_param_handler = blf.parameters_handler.StdParametersHandler()
        qp_param_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        qp_param_handler.set_parameter_bool("verbosity", False)

        if not self.qp_ik.set_kin_dyn(self.kindyn):
            raise RuntimeError("Failed to set kinDyn in QPFixedBaseInverseKinematics")
        if not self.qp_ik.initialize(handler=qp_param_handler):
            raise RuntimeError("Failed to initialize QPFixedBaseInverseKinematics")

        # Variables handler (robot velocity = nr_dofs + 6 for fixed base formulation)
        self.var_handler = blf.system.VariablesHandler()
        self.var_handler.add_variable("robotVelocity", self.nr_dofs + 6)

        # 3. Create Left Hand Task (SE3)
        l_hand_handler = blf.parameters_handler.StdParametersHandler()
        l_hand_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        l_hand_handler.set_parameter_string("frame_name", self.LEFT_HAND_FRAME)
        l_hand_handler.set_parameter_float("kp_linear", kp_cartesian)
        l_hand_handler.set_parameter_float("kp_angular", kp_angular)

        self.l_hand_task = blf.ik.SE3Task()
        self.l_hand_task.set_kin_dyn(self.kindyn)
        self.l_hand_task.initialize(param_handler=l_hand_handler)
        self.l_hand_task.set_variables_handler(variables_handler=self.var_handler)
        self.qp_ik.add_task(task=self.l_hand_task, task_name="left_hand_task", priority=1, weight=[weight_hands] * 6)

        # 4. Create Right Hand Task (SE3)
        r_hand_handler = blf.parameters_handler.StdParametersHandler()
        r_hand_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        r_hand_handler.set_parameter_string("frame_name", self.RIGHT_HAND_FRAME)
        r_hand_handler.set_parameter_float("kp_linear", kp_cartesian)
        r_hand_handler.set_parameter_float("kp_angular", kp_angular)

        self.r_hand_task = blf.ik.SE3Task()
        self.r_hand_task.set_kin_dyn(self.kindyn)
        self.r_hand_task.initialize(param_handler=r_hand_handler)
        self.r_hand_task.set_variables_handler(variables_handler=self.var_handler)
        self.qp_ik.add_task(task=self.r_hand_task, task_name="right_hand_task", priority=1, weight=[weight_hands] * 6)

        # 5. Create Head Gaze Task (SO3)
        head_handler = blf.parameters_handler.StdParametersHandler()
        head_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        head_handler.set_parameter_string("frame_name", self.HEAD_FRAME)
        head_handler.set_parameter_float("kp_angular", kp_angular)

        self.head_task = blf.ik.SO3Task()
        self.head_task.set_kin_dyn(self.kindyn)
        self.head_task.initialize(param_handler=head_handler)
        self.head_task.set_variables_handler(variables_handler=self.var_handler)
        self.qp_ik.add_task(task=self.head_task, task_name="head_gaze_task", priority=1, weight=[weight_head] * 3)

        # 6. Joint Position Limits (Priority 0 - Hard constraint)
        joint_limits_handler = blf.parameters_handler.StdParametersHandler()
        joint_limits_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        joint_limits_handler.set_parameter_vector_float("klim", [0.5] * self.nr_dofs)
        joint_limits_handler.set_parameter_bool("use_model_limits", True)
        joint_limits_handler.set_parameter_datetime("sampling_time", self.dT)

        self.joint_limits_task = blf.ik.JointLimitsTask()
        self.joint_limits_task.set_kin_dyn(self.kindyn)
        self.joint_limits_task.initialize(param_handler=joint_limits_handler)
        self.joint_limits_task.set_variables_handler(variables_handler=self.var_handler)
        self.qp_ik.add_task(task=self.joint_limits_task, task_name="joint_limits_task", priority=0)

        # 7. Joint Velocity Limits (Priority 0 - Hard constraint)
        vel_limits_handler = blf.parameters_handler.StdParametersHandler()
        vel_limits_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        vel_limits_handler.set_parameter_vector_float("upper_limits", [max_joint_velocity] * self.nr_dofs)
        vel_limits_handler.set_parameter_vector_float("lower_limits", [-max_joint_velocity] * self.nr_dofs)

        self.vel_limits_task = blf.ik.JointVelocityLimitsTask()
        self.vel_limits_task.set_kin_dyn(self.kindyn)
        self.vel_limits_task.initialize(param_handler=vel_limits_handler)
        self.vel_limits_task.set_variables_handler(variables_handler=self.var_handler)
        self.qp_ik.add_task(task=self.vel_limits_task, task_name="joint_velocity_limits_task", priority=0)

        # 8. Posture Regularization (Priority 1 - Soft constraint in null-space)
        posture_handler = blf.parameters_handler.StdParametersHandler()
        posture_handler.set_parameter_string("robot_velocity_variable_name", "robotVelocity")
        posture_handler.set_parameter_vector_float("kp", [2.0] * self.nr_dofs)

        self.posture_task = blf.ik.JointTrackingTask()
        self.posture_task.set_kin_dyn(self.kindyn)
        self.posture_task.initialize(param_handler=posture_handler)
        self.posture_task.set_variables_handler(variables_handler=self.var_handler)
        if weight_posture is not None:
            posture_weights = [float(weight_posture)] * self.nr_dofs
        else:
            # 3 Torso, 3 Neck, 7 Left Arm, 7 Right Arm
            posture_weights = (
                [float(weight_posture_torso)] * 3
                + [float(weight_posture_neck)] * 3
                + [float(weight_posture_arms)] * 14
            )

        self.qp_ik.add_task(
            task=self.posture_task,
            task_name="posture_task",
            priority=1,
            weight=posture_weights,
        )

        # Finalize QP
        if not self.qp_ik.finalize(handler=self.var_handler):
            raise RuntimeError("Failed to finalize QPFixedBaseInverseKinematics")

        self.posture_ref = self.DEFAULT_HOME_POSTURE.copy()
        self.posture_task.set_set_point(joint_position=self.posture_ref)

        # Initialize default task setpoints to current initial poses
        pos_l0, rot_l0 = self.get_current_left_hand_pose()
        pos_r0, rot_r0 = self.get_current_right_hand_pose()
        rot_h0 = self.get_current_head_orientation()
        self.set_desired_left_hand(pos_l0, rot_l0)
        self.set_desired_right_hand(pos_r0, rot_r0)
        self.set_desired_head_orientation(rot_h0)

    def set_base_pose(self, position: np.ndarray, rotation_matrix: np.ndarray):
        """Set world-to-root_link homogeneous transform (defaults to I_4 for root_link frame)."""
        self.base_pose = np.eye(4, dtype=np.float64)
        self.base_pose[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64)
        self.base_pose[:3, 3] = np.asarray(position, dtype=np.float64)
        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)

    def update_robot_state(self, joint_positions: np.ndarray, joint_velocities: Optional[np.ndarray] = None):
        """Update robot state in KinDynComputations."""
        self.s = np.asarray(joint_positions, dtype=np.float64)
        if joint_velocities is not None:
            self.s_dot = np.asarray(joint_velocities, dtype=np.float64)
        else:
            self.s_dot.fill(0.0)

        self.kindyn.setRobotState(self.base_pose, self.s, self.base_vel, self.s_dot, self.gravity)

    def get_current_frame_pose(self, frame_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Returns position (3,) and rotation matrix (3,3) for any frame."""
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

    def set_desired_left_hand(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        feedforward_velocity: Optional[np.ndarray] = None,
    ):
        """Set 6D SE(3) setpoint for the left hand."""
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_H_l_hand = manif.SE3(position=pos, quaternion=quat)

        if feedforward_velocity is not None:
            ff_vel = np.asarray(feedforward_velocity, dtype=np.float64).flatten()
            if len(ff_vel) == 3:
                ff_vel_6d = np.zeros(6, dtype=np.float64)
                ff_vel_6d[:3] = ff_vel
            elif len(ff_vel) == 6:
                ff_vel_6d = ff_vel
            else:
                raise ValueError(f"feedforward_velocity must have length 3 or 6, got {len(ff_vel)}")
            tangent_vel = manif.SE3Tangent(ff_vel_6d)
        else:
            tangent_vel = manif.SE3Tangent.Zero()

        self.l_hand_task.set_set_point(I_H_F=I_H_l_hand, mixed_velocity=tangent_vel)

    def set_desired_right_hand(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        feedforward_velocity: Optional[np.ndarray] = None,
    ):
        """Set 6D SE(3) setpoint for the right hand."""
        pos = np.asarray(position, dtype=np.float64)
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_H_r_hand = manif.SE3(position=pos, quaternion=quat)

        if feedforward_velocity is not None:
            ff_vel = np.asarray(feedforward_velocity, dtype=np.float64).flatten()
            if len(ff_vel) == 3:
                ff_vel_6d = np.zeros(6, dtype=np.float64)
                ff_vel_6d[:3] = ff_vel
            elif len(ff_vel) == 6:
                ff_vel_6d = ff_vel
            else:
                raise ValueError(f"feedforward_velocity must have length 3 or 6, got {len(ff_vel)}")
            tangent_vel = manif.SE3Tangent(ff_vel_6d)
        else:
            tangent_vel = manif.SE3Tangent.Zero()

        self.r_hand_task.set_set_point(I_H_F=I_H_r_hand, mixed_velocity=tangent_vel)

    def set_desired_head_orientation(
        self,
        rotation_matrix: np.ndarray,
        feedforward_angular_velocity: Optional[np.ndarray] = None,
    ):
        """Set 3D SO(3) setpoint for head gaze orientation."""
        rot = np.asarray(rotation_matrix, dtype=np.float64)
        quat = rotmat_to_quat(rot)
        I_R_head = manif.SO3(quaternion=quat)

        if feedforward_angular_velocity is not None:
            w = np.asarray(feedforward_angular_velocity, dtype=np.float64).flatten()
            tangent_ang_vel = manif.SO3Tangent(w)
        else:
            tangent_ang_vel = manif.SO3Tangent.Zero()

        self.head_task.set_set_point(I_R_F=I_R_head, angular_velocity=tangent_ang_vel)

    def set_desired_head_quaternion(
        self,
        quat_wxyz: np.ndarray,
        feedforward_angular_velocity: Optional[np.ndarray] = None,
    ):
        """Set 3D SO(3) setpoint for head orientation from quaternion [w, x, y, z]."""
        self.set_desired_head_orientation(
            quat_wxyz_to_rotmat(quat_wxyz),
            feedforward_angular_velocity=feedforward_angular_velocity,
        )

    def set_posture_reference(self, joint_positions: np.ndarray):
        """Set null-space resting posture setpoint."""
        self.posture_ref = np.asarray(joint_positions, dtype=np.float64)
        self.posture_task.set_set_point(joint_position=self.posture_ref)

    def step(self) -> np.ndarray:
        """Solve QP and return optimal joint velocities (20,)."""
        if not self.qp_ik.advance():
            raise RuntimeError("QPFixedBaseInverseKinematics solver failed to advance.")

        if not self.qp_ik.is_output_valid():
            raise RuntimeError("QP output is marked as invalid.")

        output = self.qp_ik.get_output()
        return np.asarray(output.joint_velocity, dtype=np.float64)

