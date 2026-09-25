"""
IK Solver for MuJoCo iCub Teleoperation.
==========================================

Arm IK:
    Uses iKinIpOptMin (from utils/ikin_ik.py) — the same constrained
    optimisation solver used by cartesian_controller.py.  Shoulder coupling
    constraints and real-robot joint limits are handled natively by iKin.

    The solver produces a **goal joint configuration**.  A minimum-jerk
    interpolation smoothly drives ``data.ctrl`` toward that goal, which
    is why movements feel much more natural than raw differential IK.

Gaze (head) IK:
    Uses a lightweight differential IK (MuJoCo Jacobian pseudoinverse)
    for the 3 neck joints — iKin does not model the neck.

Requirements:
    conda activate icubenv   (needs yarp + icub Python bindings)
"""

import numpy as np
import mujoco

# ---------------------------------------------------------------------------
# iKin-based arm IK  (requires yarp + icub)
# ---------------------------------------------------------------------------
try:
    import yarp
    import icub
    from pyquaternion import Quaternion as PyQuaternion
    IKIN_AVAILABLE = True
except ImportError:
    IKIN_AVAILABLE = False


# ===========================================================================
#                          iKin Arm Solver
# ===========================================================================

class IKinArmSolver:
    """
    Wrapper around iKinIpOptMin for one arm (left or right).

    Produces a goal joint configuration given a target EE pose.  The caller
    is responsible for interpolating ``data.ctrl`` toward that goal.
    """

    def __init__(self, side="right", joints_to_control=None,
                 real_robot_limits=True, limit_torso_pitch=True,
                 waist_pos=None, waist_rot_mat=None):
        """
        Args:
            side: "right" or "left"
            joints_to_control: list of joint names (e.g. RIGHT_ARM_JOINTS)
            real_robot_limits: apply tighter limits from the real robot
            limit_torso_pitch: cap torso_pitch max at ~28.6 deg (0.5 rad)
            waist_pos: [3] position of the waist body in MuJoCo world
            waist_rot_mat: [3,3] rotation matrix of waist in MuJoCo world
        """
        if not IKIN_AVAILABLE:
            raise RuntimeError(
                "iKin not available. Run:  conda activate icubenv")

        self.side = side
        self.joints_to_control = list(joints_to_control or [])

        # ----- Build iKin chain -----
        arm_str = f"{side}_v2.5"
        self.arm_chain = icub.iCubArm(arm_str)
        icub.iCubAdditionalArmConstraints(self.arm_chain)

        # Release torso links if controlled
        if 'torso_pitch' in self.joints_to_control:
            self.arm_chain.releaseLink(0)
        if 'torso_roll' in self.joints_to_control:
            self.arm_chain.releaseLink(1)
        if 'torso_yaw' in self.joints_to_control:
            self.arm_chain.releaseLink(2)

        self.chain = icub.iKinChain(self.arm_chain)

        if real_robot_limits:
            self._set_real_limits()
        if limit_torso_pitch and 'torso_pitch' in self.joints_to_control:
            self.chain.access(0).setMax(0.5)

        # Create solver
        self.solver = icub.iKinIpOptMin(
            self.chain, 0, 1e-2, 1e-2, 5000, verbose=0)

        self.prev_sol = None

        # Number of IK joints (torso subset + 7 arm joints)
        self.n_ik_joints = len(self.joints_to_control)

        # Frame transform: MuJoCo world → iKin (waist-local) frame
        self._waist_pos = np.array(waist_pos if waist_pos is not None
                                   else [0.0, 0.0, 0.0])
        self._waist_R = np.array(waist_rot_mat if waist_rot_mat is not None
                                  else np.eye(3))
        # Pre-compute inverse rotation quaternion for orientation transform
        self._waist_q_inv = PyQuaternion(matrix=self._waist_R).inverse

        # Track blocked (uncontrolled) torso links for setBlockingValue
        self._blocked_torso = []
        for name, link_idx in [('torso_pitch', 0),
                                ('torso_roll', 1),
                                ('torso_yaw', 2)]:
            if name not in self.joints_to_control:
                self._blocked_torso.append((link_idx, name))

        # Default rest pose (comfortable angles: shoulder down, elbow bent)
        # Order: torso (3) + shoulder (3) + elbow (1) + wrist (3)
        # Normalized to 0 for most, but elbow slightly bent often helps avoiding singularities
        # Let's use a zero vector as a base "comfy" pose, but we can tune it.
        # Actually, simply pulling towards 0.0 for all joints is often "good enough" for iCub arms
        # except maybe elbow which prefers ~0.5-1.0 rad.
        self._rest_pose = yarp.Vector(self.n_ik_joints)
        for i in range(self.n_ik_joints):
             self._rest_pose.set(i, 0.0)
        
        # Indices depend on whether torso joints are included
        has_torso = 'torso_pitch' in self.joints_to_control
        torso_offset = 3 if has_torso else 0

        # shoulder_pitch: negative = arm forward/down (natural position)
        shoulder_pitch_idx = torso_offset + 0
        if shoulder_pitch_idx < self.n_ik_joints:
            self._rest_pose.set(shoulder_pitch_idx, -0.5)  # ~-28 deg (arm forward)

        # shoulder_roll: moderate value keeps elbow pointing down
        shoulder_roll_idx = torso_offset + 1
        if shoulder_roll_idx < self.n_ik_joints:
            self._rest_pose.set(shoulder_roll_idx, 0.5)   # ~28 deg

        # elbow: bent to avoid singularities and keep elbow down
        elbow_idx = torso_offset + 3
        if elbow_idx < self.n_ik_joints:
            self._rest_pose.set(elbow_idx, 1.0)  # ~57 deg bent

    def _set_real_limits(self):
        """Set tighter joint limits matching the real iCub."""
        deg2rad = np.pi / 180
        idx = 0
        
        # --- Torso Limits (Restricted to Pitch only) ---
        if 'torso_pitch' in self.joints_to_control:
            # Allow forward lean (0 to ~40 deg) and slight backward (-10 deg)
            self.chain.access(idx).setMin(-10.0 * deg2rad)
            self.chain.access(idx).setMax(10.0 * deg2rad)
            idx += 1
        
        if 'torso_roll' in self.joints_to_control:
            # Lock Roll
            self.chain.access(idx).setMin(0.0)
            self.chain.access(idx).setMax(0.0)
            idx += 1
            
        if 'torso_yaw' in self.joints_to_control:
            # Lock Yaw
            self.chain.access(idx).setMin(-10.0 * deg2rad)
            self.chain.access(idx).setMax(10.0 * deg2rad)
            idx += 1

        for jname in ("r_shoulder_yaw", "l_shoulder_yaw"):
            if jname in self.joints_to_control:
                j_idx = self.joints_to_control.index(jname)
                self.chain.access(j_idx).setMin(-22.0 * deg2rad)
                self.chain.access(j_idx).setMax(70.0 * deg2rad)
                break

    def solve(self, target_pos, target_quat_wxyz, current_joint_values,
              torso_values=None):
        """
        Compute the goal joint configuration for a target EE pose.

        Args:
            target_pos: desired EE position [3] (MuJoCo world frame)
            target_quat_wxyz: desired quaternion [4] (w,x,y,z) world frame
            current_joint_values: current joint angles [n_ik_joints] in rad
            torso_values: dict {name: rad} for torso joints NOT in this
                          solver's control set (needed when torso is
                          controlled by the other arm).

        Returns:
            (solution_rad, solved): tuple of (np.array, bool)
                solution_rad has shape [n_ik_joints]
        """
        # Set blocking values for non-controlled torso joints
        if torso_values:
            for link_idx, name in self._blocked_torso:
                if name in torso_values:
                    self.chain.setBlockingValue(link_idx, torso_values[name])

        # --- Transform target from MuJoCo world → iKin (waist-local) frame ---
        pos_local = self._waist_R.T @ (np.asarray(target_pos) - self._waist_pos)

        q_world = PyQuaternion(w=target_quat_wxyz[0], x=target_quat_wxyz[1],
                               y=target_quat_wxyz[2], z=target_quat_wxyz[3])
        q_local = self._waist_q_inv * q_world
        axis_angle = np.append(q_local.axis, q_local.angle)

        # Build target vector  [px, py, pz, ax, ay, az, angle]
        target = yarp.Vector(np.concatenate((pos_local, axis_angle)))

        # Current joint values as yarp vector
        current_yarp = yarp.Vector(current_joint_values.astype(np.float64))

        n = self.n_ik_joints
        
        # Optimization weights
        # Task 1 (End Effector): Implicitly handled by solver constraints/objective
        # Task 2 (Optional constraint): Weight 0.0 (unused)
        # Task 3 (Posture/Regularization): Weight 0.1 - 1.0
        # We use a lower weight for posture so it doesn't fight the primary task too hard
        # But high enough to influence null-space motion.
        
        # REGULARIZATION STRATEGY:
        # Instead of staying close to "previous solution" (which drifts),
        # we try to stay close to "rest pose" (which fixes elbow/shoulder behavior).
        # We use current_yarp as initialization for speed/solver stability,
        # but the COST FUNCTION (arg #8/9) targets the REST POSE.
        
        # Args:
        # 1. current_q
        # 2. target_pose
        # 3. weight_task2
        # 4. ref_task2
        # 5. mask_task2
        # 6. weight_task3 (posture)
        # 7. ref_task3 (desired posture)
        # 8. mask_task3 (weights per joint)
        
        # We want to use the previous solution as the *seed* (first arg)
        # but the rest pose as the *target* for the secondary task (7th arg).
        
        seed_q = self.prev_sol if self.prev_sol is not None else current_yarp

        solution_yarp = self.solver.solve(
            seed_q,                          # Initialization seed
            target,
            0.0,
            yarp.Vector([0.0]),
            yarp.Vector([0.0]),
            1e-4,                            # Low weight for posture (just a hint)
            self._rest_pose,                 # Target: Rest pose (elbow down)
            yarp.Vector(np.ones(n)),         # All joints affected
        )

        self.prev_sol = solution_yarp

        # Extract solution
        solution = np.array([solution_yarp.get(i)
                             for i in range(solution_yarp.getListSize())])

        # Check if EE reached the target (in iKin frame)
        predicted_pose = self.chain.EndEffPose()
        predicted_pos = np.array([predicted_pose.get(i) for i in range(3)])
        solved = np.linalg.norm(predicted_pos - pos_local) < 0.03

        return solution, solved

    def reset(self):
        """Reset solver state (call when teleop starts or after go-home)."""
        self.prev_sol = None


# ===========================================================================
#                       Gaze (differential IK for neck)
# ===========================================================================

class GazeIKSolver:
    """
    Lightweight differential IK for the 3 neck joints using MuJoCo Jacobians.

    Points the head's z-axis (forward direction) toward a target point.
    """

    def __init__(self, model, data, damping=1e-4, integration_dt=0.1,
                 max_angvel=2.0):
        self.model = model
        self.data = data
        self.damping = damping
        self.integration_dt = integration_dt
        self.max_angvel = max_angvel
        self._diag3 = damping * np.eye(3)

    def solve(self, head_body_name, target_pos, neck_joint_names):
        """
        Compute neck joint ctrl deltas to point the head toward target_pos.

        Returns:
            dict: {joint_name: desired_ctrl_value}
        """
        head_body_id = self.model.body(head_body_name).id
        head_pos = self.data.xpos[head_body_id]
        head_mat = self.data.xmat[head_body_id].reshape(3, 3)

        # iCub head z-axis = forward direction
        current_forward = head_mat[:, 2]

        direction = target_pos - head_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return {}
        desired_forward = direction / dist

        # Angular error
        axis = np.cross(current_forward, desired_forward)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(current_forward, desired_forward)
        if sin_angle < 1e-8:
            return {}
        axis /= sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        error_ori = axis * angle

        # Neck joint DoF indices
        joint_ids = [self.model.joint(n).id for n in neck_joint_names]
        dof_ids = np.array([self.model.jnt_dofadr[jid] for jid in joint_ids])

        # Rotational Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, head_body_id)
        jac = jacr[:, dof_ids]

        dq = jac.T @ np.linalg.solve(jac @ jac.T + self._diag3, error_ori)

        # Clamp velocity
        if self.max_angvel > 0:
            dq_abs = np.abs(dq).max()
            if dq_abs > self.max_angvel:
                dq *= self.max_angvel / dq_abs

        # Integrate from current ctrl (preserves gravity offset)
        result = {}
        for i, (jname, jid) in enumerate(zip(neck_joint_names, joint_ids)):
            act_id = self.model.actuator(jname).id
            desired = self.data.ctrl[act_id] + dq[i] * self.integration_dt
            desired = np.clip(desired,
                              self.model.jnt_range[jid, 0],
                              self.model.jnt_range[jid, 1])
            result[jname] = desired
        return result


# ===========================================================================
#                  Minimum-jerk interpolation helper
# ===========================================================================

def min_jerk_scalar(tau):
    """Minimum-jerk interpolation factor for normalised time t in [0, 1].

    Returns s in [0, 1] such that  q(t) = q0 + (q1 - q0) * s.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5
