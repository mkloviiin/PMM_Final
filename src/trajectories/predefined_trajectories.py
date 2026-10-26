#!/usr/bin/env python3
"""
Predefined smooth bimanual and gaze trajectories for iCub testing.
Includes smooth ramp-in blending, forward reaching, circular motions,
and head pan-tilt gaze scanning with analytical feedforward velocities.
"""

import numpy as np
import manifpy as manif
from typing import Tuple, Dict


def euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert RPY angles (radians) to 3x3 rotation matrix (Z-Y-X convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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


def world_rpy_kinematics(
    rpy: np.ndarray,
    drpy: np.ndarray,
    ddrpy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3x3 rotation matrix R_rel (Z-Y-X extrinsic / world-frame RPY),
    exact world-frame angular velocity omega (3,), and angular acceleration domega (3,)
    such that d/dt (R_rel @ R0) = [omega]_x (R_rel @ R0).
    """
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    droll, dpitch, dyaw = float(drpy[0]), float(drpy[1]), float(drpy[2])

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R_rel = Rz @ Ry @ Rx

    # World-frame angular velocity mapping E(pitch, yaw) for Z-Y-X Euler angles
    E = np.array([
        [cy * cp, -sy, 0.0],
        [sy * cp,  cy, 0.0],
        [-sp,      0.0, 1.0],
    ])
    dE = np.array([
        [-sy * dyaw * cp - cy * sp * dpitch, -cy * dyaw, 0.0],
        [ cy * dyaw * cp - sy * sp * dpitch, -sy * dyaw, 0.0],
        [-cp * dpitch,                       0.0,        0.0],
    ])

    omega = E @ drpy
    domega = E @ ddrpy + dE @ drpy
    return R_rel, omega, domega


def slerp_rotmat(
    R_start: np.ndarray,
    R_target: np.ndarray,
    s: float,
    s_dot: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Geodesic SO(3) spherical linear interpolation (SLERP) using left-trivialized
    inertial-frame tangent vector so that R_interp(1.0) == R_target and ang_vel
    is expressed in the inertial (world) frame (MIXED_REPRESENTATION).
    Returns: (R_interp (3,3), ang_vel (3,))
    """
    q1 = rotmat_to_quat(R_start)
    q2 = rotmat_to_quat(R_target)
    so3_1 = manif.SO3(quaternion=q1)
    so3_2 = manif.SO3(quaternion=q2)
    tangent_diff = (so3_2.compose(so3_1.inverse())).log()
    so3_interp = so3_1.lplus(tangent_diff * s)
    R_interp = so3_interp.rotation()
    ang_vel = tangent_diff.coeffs() * s_dot
    return R_interp, ang_vel


def smooth_ramp(t: float, duration: float = 2.0) -> Tuple[float, float]:
    """
    Quintic polynomial ramp s(t) from 0 to 1 with zero velocity and acceleration at ends.
    Returns: (s, s_dot)
    """
    s, s_dot, _ = smooth_ramp_with_acc(t, duration)
    return s, s_dot


def smooth_ramp_with_acc(t: float, duration: float = 2.0) -> Tuple[float, float, float]:
    """
    Quintic polynomial ramp s(t) from 0 to 1 with first and second time derivatives.
    Returns: (s, s_dot, s_ddot)
    """
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    if t >= duration:
        return 1.0, 0.0, 0.0

    tau = t / duration
    s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
    s_dot = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / duration
    s_ddot = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (duration ** 2)
    return s, s_dot, s_ddot


class BimanualGazeTrajectoryGenerator:
    """
    Generates coordinated trajectories for:
    - Left Hand: 6D pose (position + orientation) + twist (+ 6D acceleration)
    - Right Hand: 6D pose + twist (+ 6D acceleration)
    - Head Gaze: 3D orientation (SO3) + angular velocity (+ 3D angular acceleration)
    """

    def __init__(
        self,
        initial_left_pos: np.ndarray,
        initial_left_rot: np.ndarray,
        initial_right_pos: np.ndarray,
        initial_right_rot: np.ndarray,
        initial_head_rot: np.ndarray,
        reach_amplitude_x: float = 0.06,  # 6 cm forward/backward
        reach_amplitude_z: float = 0.04,  # 4 cm vertical oscillation
        frequency_arms: float = 0.25,     # 4 second period
        hand_roll_amplitude: float = 0.20,   # ~11.5 deg wrist pronosupination oscillation
        hand_pitch_amplitude: float = 0.18,  # ~10 deg wrist pitch oscillation
        hand_yaw_amplitude: float = 0.12,    # ~7 deg wrist yaw oscillation
        gaze_yaw_amplitude: float = 0.25, # ~14 deg horizontal scan
        gaze_pitch_amplitude: float = 0.15, # ~8 deg vertical scan
        frequency_gaze: float = 0.2,      # 5 second period
        ramp_duration: float = 2.5,       # 2.5s smooth start from rest
    ):
        self.p_l0 = np.copy(initial_left_pos)
        self.R_l0 = np.copy(initial_left_rot)
        self.p_r0 = np.copy(initial_right_pos)
        self.R_r0 = np.copy(initial_right_rot)
        self.R_h0 = np.copy(initial_head_rot)

        self.amp_x = reach_amplitude_x
        self.amp_z = reach_amplitude_z
        self.omega_arms = 2.0 * np.pi * frequency_arms

        self.amp_roll_h = hand_roll_amplitude
        self.amp_pitch_h = hand_pitch_amplitude
        self.amp_yaw_h = hand_yaw_amplitude

        self.amp_yaw = gaze_yaw_amplitude
        self.amp_pitch = gaze_pitch_amplitude
        self.omega_gaze = 2.0 * np.pi * frequency_gaze

        self.ramp_duration = ramp_duration

    def evaluate_full(self, t: float) -> Dict[str, Tuple[np.ndarray, ...]]:
        """
        Evaluate trajectory at time t including analytical second derivatives (accelerations).
        Returns dictionary containing:
        - 'left_hand':  (pos (3,), rot (3,3), twist_6d (6,), acc_6d (6,))
        - 'right_hand': (pos (3,), rot (3,3), twist_6d (6,), acc_6d (6,))
        - 'head_gaze':  (rot (3,3), ang_vel (3,), ang_acc (3,))
        """
        s, s_dot, s_ddot = smooth_ramp_with_acc(t, self.ramp_duration)
        w = self.omega_arms
        w2 = w * w

        # 1. Left Hand (Frontal Reach in -X & Dynamic Rotation)
        c_l = np.cos(w * t)
        s_l = np.sin(w * t)

        delta_l = np.array([-self.amp_x * (1.0 - c_l) * 0.5, 0.0, self.amp_z * s_l])
        d_delta_l = np.array([-self.amp_x * w * s_l * 0.5, 0.0, self.amp_z * w * c_l])
        dd_delta_l = np.array([-self.amp_x * w2 * c_l * 0.5, 0.0, -self.amp_z * w2 * s_l])

        pos_l = self.p_l0 + s * delta_l
        vel_l_lin = s * d_delta_l + s_dot * delta_l
        acc_l_lin = s * dd_delta_l + 2.0 * s_dot * d_delta_l + s_ddot * delta_l

        # Hand rotations (pronosupination/roll, pitch, yaw in world frame)
        raw_rpy_l = np.array([
            self.amp_roll_h * s_l,
            self.amp_pitch_h * (1.0 - c_l) * 0.5,
            self.amp_yaw_h * s_l,
        ])
        d_raw_rpy_l = np.array([
            self.amp_roll_h * w * c_l,
            self.amp_pitch_h * w * s_l * 0.5,
            self.amp_yaw_h * w * c_l,
        ])
        dd_raw_rpy_l = np.array([
            -self.amp_roll_h * w2 * s_l,
            self.amp_pitch_h * w2 * c_l * 0.5,
            -self.amp_yaw_h * w2 * s_l,
        ])

        rpy_l = s * raw_rpy_l
        drpy_l = s * d_raw_rpy_l + s_dot * raw_rpy_l
        ddrpy_l = s * dd_raw_rpy_l + 2.0 * s_dot * d_raw_rpy_l + s_ddot * raw_rpy_l

        R_rel_l, omega_l, domega_l = world_rpy_kinematics(rpy_l, drpy_l, ddrpy_l)
        rot_l = R_rel_l @ self.R_l0
        twist_l = np.concatenate([vel_l_lin, omega_l])
        acc_l = np.concatenate([acc_l_lin, domega_l])

        # 2. Right Hand (Mirrored / Alternated Reach in -X & Rotation)
        phase_r = np.pi * 0.5  # 90 deg phase offset for asymmetric bimanual reach
        c_r = np.cos(w * t + phase_r)
        s_r = np.sin(w * t + phase_r)

        delta_r = np.array([-self.amp_x * (1.0 - c_r) * 0.5, 0.0, self.amp_z * s_r])
        d_delta_r = np.array([-self.amp_x * w * s_r * 0.5, 0.0, self.amp_z * w * c_r])
        dd_delta_r = np.array([-self.amp_x * w2 * c_r * 0.5, 0.0, -self.amp_z * w2 * s_r])

        pos_r = self.p_r0 + s * delta_r
        vel_r_lin = s * d_delta_r + s_dot * delta_r
        acc_r_lin = s * dd_delta_r + 2.0 * s_dot * d_delta_r + s_ddot * delta_r

        raw_rpy_r = np.array([
            -self.amp_roll_h * s_r,
            self.amp_pitch_h * (1.0 - c_r) * 0.5,
            -self.amp_yaw_h * s_r,
        ])
        d_raw_rpy_r = np.array([
            -self.amp_roll_h * w * c_r,
            self.amp_pitch_h * w * s_r * 0.5,
            -self.amp_yaw_h * w * c_r,
        ])
        dd_raw_rpy_r = np.array([
            self.amp_roll_h * w2 * s_r,
            self.amp_pitch_h * w2 * c_r * 0.5,
            self.amp_yaw_h * w2 * s_r,
        ])

        rpy_r = s * raw_rpy_r
        drpy_r = s * d_raw_rpy_r + s_dot * raw_rpy_r
        ddrpy_r = s * dd_raw_rpy_r + 2.0 * s_dot * d_raw_rpy_r + s_ddot * raw_rpy_r

        R_rel_r, omega_r, domega_r = world_rpy_kinematics(rpy_r, drpy_r, ddrpy_r)
        rot_r = R_rel_r @ self.R_r0
        twist_r = np.concatenate([vel_r_lin, omega_r])
        acc_r = np.concatenate([acc_r_lin, domega_r])

        # 3. Head Gaze (Smooth Yaw & Pitch scanning in world frame)
        wg = self.omega_gaze
        wg2 = wg * wg
        raw_gaze = np.array([
            0.0,
            self.amp_pitch * np.sin(2.0 * wg * t),
            self.amp_yaw * np.sin(wg * t),
        ])
        d_raw_gaze = np.array([
            0.0,
            self.amp_pitch * 2.0 * wg * np.cos(2.0 * wg * t),
            self.amp_yaw * wg * np.cos(wg * t),
        ])
        dd_raw_gaze = np.array([
            0.0,
            -self.amp_pitch * 4.0 * wg2 * np.sin(2.0 * wg * t),
            -self.amp_yaw * wg2 * np.sin(wg * t),
        ])

        gaze_rpy = s * raw_gaze
        dgaze_rpy = s * d_raw_gaze + s_dot * raw_gaze
        ddgaze_rpy = s * dd_raw_gaze + 2.0 * s_dot * d_raw_gaze + s_ddot * raw_gaze

        R_rel_h, ang_vel_head, ang_acc_head = world_rpy_kinematics(gaze_rpy, dgaze_rpy, ddgaze_rpy)
        rot_head = R_rel_h @ self.R_h0

        return {
            "left_hand": (pos_l, rot_l, twist_l, acc_l),
            "right_hand": (pos_r, rot_r, twist_r, acc_r),
            "head_gaze": (rot_head, ang_vel_head, ang_acc_head),
        }

    def evaluate(self, t: float) -> Dict[str, Tuple[np.ndarray, ...]]:
        """
        Evaluate trajectory at time t.
        Returns dictionary containing:
        - 'left_hand':  (pos (3,), rot (3,3), twist_6d (6,))
        - 'right_hand': (pos (3,), rot (3,3), twist_6d (6,))
        - 'head_gaze':  (rot (3,3), ang_vel (3,))
        """
        full = self.evaluate_full(t)
        pos_l, rot_l, twist_l, _ = full["left_hand"]
        pos_r, rot_r, twist_r, _ = full["right_hand"]
        rot_head, ang_vel_head, _ = full["head_gaze"]
        return {
            "left_hand": (pos_l, rot_l, twist_l),
            "right_hand": (pos_r, rot_r, twist_r),
            "head_gaze": (rot_head, ang_vel_head),
        }

