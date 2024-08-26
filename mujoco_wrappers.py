"""
    Wrapper classes for accessing and setting the state of objects in a mujoco simulation.

    mail@kaiploeger.net
"""

import mujoco as mj
import numpy as np


class MjObject:
    def __init__(self, m: mj.MjModel, d: mj.MjData,
                 free_joint_name, body_name, mocap_name=None):
        """ Simple wrapper around mujoco model and data to access and set the state of an object with a free joint. """
        self.model = m
        self.data = d
        self.free_joint = d.joint(free_joint_name)
        self.body = d.body(body_name)
        self.mocap_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, mocap_name) if mocap_name is not None else None
        self.mocap_id = self.model.body_mocapid[self.mocap_body_id] if self.mocap_body_id is not None else None

    @property
    def pos(self) -> np.ndarray:
        return self.free_joint.qpos[:3]

    @pos.setter
    def pos(self, x: np.ndarray):
        self.free_joint.qpos[:3] = x

    @property
    def vel(self) -> np.ndarray:
        return self.free_joint.qvel[:3]

    @vel.setter
    def vel(self, dx):
        self.free_joint.qvel[:3] = dx

    @property
    def quat(self) -> np.ndarray:
        return self.free_joint.qpos[3:]
    
    @quat.setter
    def quat(self, q):
        self.free_joint.qpos[3:] = q

    @property
    def R(self) -> np.ndarray:
        rotation_matrix = np.zeros(9)
        quat = self.quat
        mj.mju_quat2Mat(rotation_matrix, quat)
        return rotation_matrix.reshape(3, 3)
    
    @R.setter
    def R(self, R):
        assert R.shape == (3, 3)
        assert np.isclose(np.linalg.det(R), 1)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, R.flatten())
        self.quat = quat

    @property
    def angvel(self) -> np.ndarray:
        return self.free_joint.qvel[3:]
    
    @angvel.setter
    def angvel(self, w):
        self.free_joint.qvel[3:] = w

    @property
    def mass(self):
        return self.model.body_mass[self.body.id]
    
    def apply_force_torque(self, force=None, torque=None, point=None):
        force = np.zeros(3) if force is None else force
        torque = np.zeros(3) if torque is None else torque
        if point is None: # default to appling at center of mass
            self.data.qfrc_applied[(self.body.id-1)*6:self.body.id*6] = np.concatenate([force, torque])
        else:
            mj.mj_applyFT(self.model, self.data, force, torque, point, self.body.id, self.data.qfrc_applied)

    def set_mocap(self, x):
        assert self.mocap_id is not None
        self.data.mocap_pos[self.mocap_id] = x


class Ball(MjObject):
    """ Interface for juggling balls in a mujoco model. """
    def __init__(self, m: mj.MjModel, d: mj.MjData, ball_id: int):
        self.ball_id = ball_id
        free_joint_name = 'ball'+str(ball_id)
        mocap_name = 'balls_des/ball'+str(ball_id)
        body_name = 'balls/ball'+str(ball_id)
        super().__init__(m, d, free_joint_name, body_name, mocap_name)


class Hand(MjObject):
    """ Interface for an unactuated hand in a mujoco model. """
    def __init__(self, m: mj.MjModel, d: mj.MjData, free_joint_name: str, body_name: str):
        super().__init__(m, d, free_joint_name, body_name)

