"""Gravity torques of one OpenArm arm, from the MJCF (docs/teleop.md, Impedance).

The torque each joint needs to hold the arm still at a given configuration:
MuJoCo's bias force (recursive Newton-Euler) with zero velocity, so gravity
only. Joint angles and torques use the MJCF's convention, which is the real
arm's: dora-openarm takes the IK's joint angles as they are, and torques
recorded on the robot follow this model's sign on every joint
(docs/teleop.md, Impedance).
"""

from __future__ import annotations

import mujoco
import numpy as np

from dora_openarm_teleop import model_info

ARM_DOF = 7


class ArmGravity:
    """Gravity torques for one side's seven arm joints."""

    def __init__(self, side: str, scene: str = "demo") -> None:
        if side not in model_info.SIDES:
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        self.side = side
        self._model = model_info.load_model(scene)
        # Its own data: the cached model is shared, the state is not.
        self._data = mujoco.MjData(self._model)
        facts = model_info._facts(scene)
        self._qpos_adr = facts.arm_qpos_adr[side]  # joint1..7, finger1, finger2
        self._finger_ratio = facts.hulls[side][1].ratio
        joints = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{j}")
            for j in range(1, ARM_DOF + 1)
        ]
        self._dofs = [int(self._model.jnt_dofadr[j]) for j in joints]
        self._bias = np.zeros(self._model.nv)

    def torques(self, qpos: np.ndarray) -> np.ndarray:
        """Nm for joints 1-7 at ``qpos`` (joints 1-7, plus the gripper if given)."""
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.shape[0] < ARM_DOF or not np.all(np.isfinite(q)):
            raise ValueError("qpos must hold at least 7 finite joint angles")
        model, data = self._model, self._data
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[list(self._qpos_adr[:ARM_DOF])] = q[:ARM_DOF]
        if q.shape[0] > ARM_DOF:
            # The fingers' mass moves a little with the gripper angle.
            data.qpos[self._qpos_adr[7]] = q[ARM_DOF]
            data.qpos[self._qpos_adr[8]] = self._finger_ratio * q[ARM_DOF]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_rne(model, data, 0, self._bias)
        return self._bias[self._dofs].copy()
