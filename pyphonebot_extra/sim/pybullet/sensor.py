#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
import numpy as np
import pybullet as pb
from typing import List, Callable, Any

import gym


class SensorBase(ABC):
    @abstractmethod
    def sense(self) -> Any:
        """Returns the results of the sensor. This function will be called
        by sensor-users, and the return type should be defined and documented
        in the specific implementation of this base class.
        """
        raise NotImplementedError('')

    @abstractproperty
    def size(self) -> int:
        """The number of measurements sensed.
        """
        raise NotImplementedError('')

    @abstractproperty
    def space(self) -> gym.Space:
        """OpenAI gym models require inputs to for an RL agent to be `gym.Space`
        types to standardize inputs. In order to make sensors be able to feed
        directly into gym models, a space must be provided which reflects the
        bounds of the sensors range.
        """
        raise NotImplementedError('')


class BasePoseSensor(SensorBase):
    """
    Pybullet sensor for base pose.
    """

    # Pose is represented by a position (3 vector) and quaternion (4 vector)
    space = gym.spaces.Tuple(
        (gym.spaces.Box(-np.inf, np.inf, (3,),
                        dtype=np.float32),
         gym.spaces.Box(-1.0, 1.0, (4,),
                        dtype=np.float32)))

    def __init__(self):
        self.sim_id_ = -1
        self.robot_id_ = -1

    def reset(self, sim_id: int, robot_id: int):
        # Save IDs...
        self.sim_id_ = sim_id
        self.robot_id_ = robot_id

    def sense(self):
        txn, rxn = pb.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.sim_id)
        return (txn, rxn)

    @property
    def sim_id(self):
        return self.sim_id_

    @property
    def robot_id(self):
        return self.robot_id_

    @property
    def size(self):
        return 7


class JointStateSensor(SensorBase):
    """
    Pybullet sensor for joint states.
    """

    def __init__(self, joint_indices: list = None):
        # handles
        self.sim_id_ = -1
        self.robot_id_ = -1

        # params
        self.joint_indices_ = joint_indices
        self.num_joints_ = 0

        # cfg
        self.space_ = None

    def reset(self, sim_id: int, robot_id: int):
        # Save IDs...
        self.sim_id_ = sim_id
        self.robot_id_ = robot_id

        # Get relevant parameters.
        if self.joint_indices_ is None:
            # FIXME(ycho): what if # of joints changes per reset?
            self.num_joints_ = pb.getNumJoints(
                self.robot_id, physicsClientId=sim_id)
            self.joint_indices_ = list(range(self.num_joints_))
        else:
            self.num_joints_ = len(self.joint_indices_)

        # FIXME(ycho): Determine `space` type based on prismatic/revolute
        jlim_lo = np.zeros(self.num_joints_)
        jlim_hi = np.zeros(self.num_joints_)
        for i, j in enumerate(self.joint_indices_):
            joint_info = pb.getJointInfo(robot_id, j, physicsClientId=sim_id)
            joint_type = joint_info[2]

            lo = joint_info[8]
            hi = joint_info[9]

            jlim_lo[i] = lo
            jlim_hi[i] = hi

        self.space_ = gym.spaces.Box(lo, hi, dtype=np.float32)

    def sense(self):
        joint_states = pb.getJointStates(
            self.robot_id, self.joint_indices_, physicsClientId=self.sim_id)
        return joint_states

    @property
    def sim_id(self):
        return self.sim_id_

    @property
    def robot_id(self):
        return self.robot_id_

    @property
    def space(self):
        return self.space_


class FlatSensorWrapper(SensorBase):
    def __init__(self, sensors: OrderedDict):
        assert isinstance(sensors, OrderedDict), 'sensors must be OrderedDict'
        self.sensors_ = sensors
        self.structure_ = OrderedDict(
            [(k, s.space) for k, s in sensors.items()])
        self.space_ = gym.spaces.flatten_space(self.structure_)

    @property
    def space(self):
        return self.space_

    @property
    def sim_id(self):
        return self.sim_id_

    @property
    def robot_id(self):
        return self.robot_id_

    def sense(self, out=None):
        return gym.spaces.flatten(
            self.structure_, {k: s.sense() for k, s in self.sensors_})


class PybulletPhonebotSensor(object):
    def __init__(self,
                 use_active_joints: bool = True,
                 use_passive_joints: bool = True,
                 use_base_position: bool = True,
                 use_base_velocity: bool = True,

                 active_joint_names: List[str] = None,
                 passive_joint_names: List[str] = None,
                 index_from_name: Callable[[str], int] = None
                 ):
        # Configure sensing state.
        self.use_aj_ = use_active_joints
        self.use_pj_ = use_passive_joints
        self.use_bp_ = use_base_position
        self.use_bv_ = use_base_velocity

        # Save joint indices.
        self.ajn_ = active_joint_names
        self.pjn_ = passive_joint_names
        self.n2i_ = index_from_name
        self.aji_ = None
        self.pji_ = None

        if (self.use_aj_ and self.ajn_ is None):
            raise ValueError(
                "Joint indices must be supplied for extracting joint states.")

        if (self.use_pj_ and self.pjn_ is None):
            raise ValueError(
                "Joint indices must be supplied for extracting joint states.")

        # Compute state size.
        state_config = self.compute_state_config()
        (self.saj_, self.sajv_, self.sajt_, self.spj_,
            self.spjv_, self.spjt_, self.sbp_, self.sbv_,
            self.size_) = state_config

        # Compute observation space ...
        lo = np.zeros(self.size_, dtype=np.float32)
        hi = np.zeros(self.size_, dtype=np.float32)

        # active joint limits
        if self.use_aj_:
            lo[self.saj_] = -np.pi / 2
            hi[self.saj_] = +np.pi / 2
            lo[self.sajv_] = -np.inf
            hi[self.sajv_] = +np.inf
            lo[self.sajt_] = -np.inf
            hi[self.sajt_] = +np.inf

        if self.use_pj_:
            lo[self.spj_] = -np.pi
            hi[self.spj_] = +np.pi
            lo[self.spjv_] = -np.inf
            hi[self.spjv_] = +np.inf
            lo[self.spjt_] = -np.inf
            hi[self.spjt_] = +np.inf

        if self.use_bp_:
            lo[self.sbp_] = -np.inf
            hi[self.sbp_] = +np.inf

        if self.use_bv_:
            lo[self.sbv_] = -np.inf
            hi[self.sbv_] = +np.inf

        self.space_ = gym.spaces.Box(
            lo, hi, shape=(self.size_,), dtype=np.float32)

    @property
    def size(self):
        return self.size_

    @property
    def slice_active_joints(self):
        return self.saj_

    @property
    def slice_passive_joints(self):
        return self.spj_

    @property
    def slice_active_joints_velocity(self):
        return self.sajv_

    @property
    def slice_passive_joints_velocity(self):
        return self.spjv_

    @property
    def slice_active_joints_torque(self):
        return self.sajt_

    @property
    def slice_passive_joints_torque(self):
        return self.spjt_

    @property
    def slice_base_position(self):
        return self.sbp_

    @property
    def slice_base_velocity(self):
        return self.sbv_

    @property
    def space(self):
        return self.space_

    def compute_state_config(self):
        n = 0
        # Active joint pos, vel, torque
        dn = self.use_aj_ * 8
        saj = np.s_[n:n + dn]
        n += dn

        dn = self.use_aj_ * 8
        sajv = np.s_[n:n + dn]
        n += dn

        dn = self.use_aj_ * 8
        sajt = np.s_[n:n + dn]
        n += dn

        # Passive joint pos, vel, torque
        dn = self.use_pj_ * 8
        spj = np.s_[n:n + dn]
        n += dn

        dn = self.use_pj_ * 8
        spjv = np.s_[n:n + dn]
        n += dn

        dn = self.use_pj_ * 8
        spjt = np.s_[n:n + dn]
        n += dn

        dn = self.use_bp_ * 7
        sbp = np.s_[n:n + dn]
        n += dn

        dn = self.use_bv_ * 6
        sbv = np.s_[n:n + dn]
        n += dn

        return (saj, sajv, sajt, spj, spjv, spjt, sbp, sbv, n)

    def sense_base_position(self, out=None):
        sim_id = self.sim_id_
        robot_id = self.robot_id_
        if out is None:
            out = np.zeros(7, dtype=np.float32)
        bx, bq = pb.getBasePositionAndOrientation(
            robot_id, physicsClientId=sim_id)
        out[0:3] = bx
        out[3:7] = bq
        return out

    def sense_base_velocity(self, out=None):
        sim_id = self.sim_id_
        robot_id = self.robot_id_
        if out is None:
            out = np.zeros(6, dtype=np.float32)
        bv, bw = pb.getBaseVelocity(robot_id, physicsClientId=sim_id)
        out[0:3] = bv
        out[3:6] = bw
        return out

    def sense_joints(self, joint_indices: np.ndarray,
                     out: np.ndarray = None) -> np.ndarray:
        """Sense the joint positions, returning joint angles.

        Args:
            joint_indices (np.ndarray): A numpy array of joint indices.
            out (np.ndarray, optional): The array to output to. If not provided,
                the shape is joint_indices.shape. Defaults to None.

        Returns:
            np.ndarray: An array of the same shape as joint indices representing
                the joint positions.
        """
        sim_id = self.sim_id_
        robot_id = self.robot_id_
        shape = np.shape(joint_indices)
        if out is None:
            out = np.zeros(shape, dtype=np.float32)
        joint_states = pb.getJointStates(
            robot_id, np.ravel(joint_indices), physicsClientId=sim_id)

        jpos = [j[0] for j in joint_states]
        out[...] = np.reshape(jpos, out.shape)
        return out

    def sense_joint_velocities(self, joint_indices: np.ndarray,
                               out: np.ndarray = None) -> np.ndarray:
        """Sense the joint velocities, returning joint angular velocities.

        Args:
            joint_indices (np.ndarray): A numpy array of joint indices.
            out (np.ndarray, optional): The array to output to. If not provided,
                the shape is joint_indices.shape. Defaults to None.

        Returns:
            np.ndarray: An array of the same shape as joint indices representing
                the joint angular velocities in radians per second.
        """
        sim_id = self.sim_id_
        robot_id = self.robot_id_
        shape = np.shape(joint_indices)
        if out is None:
            out = np.zeros(shape, dtype=np.float32)
        joint_states = pb.getJointStates(
            robot_id, np.ravel(joint_indices), physicsClientId=sim_id)

        jvel = [j[1] for j in joint_states]
        out[...] = np.reshape(jvel, out.shape)
        return out

    def sense_joint_torques(self, joint_indices: np.ndarray,
                            out: np.ndarray = None) -> np.ndarray:
        """Sense the joint velocities, returning joint torques applied by the
        motors.

        Args:
            joint_indices (np.ndarray): A numpy array of joint indices.
            out (np.ndarray, optional): The array to output to. If not provided,
                the shape is joint_indices.shape. Defaults to None.

        Returns:
            np.ndarray: An array of the same shape as joint indices representing
                the joint torques.
        """
        sim_id = self.sim_id_
        robot_id = self.robot_id_
        shape = np.shape(joint_indices)
        if out is None:
            out = np.zeros(shape, dtype=np.float32)
        joint_states = pb.getJointStates(
            robot_id, np.ravel(joint_indices), physicsClientId=sim_id)

        jtorque = [j[3] for j in joint_states]
        out[...] = np.reshape(jtorque, out.shape)
        return out

    def reset(self, sim_id: int, robot_id: int):
        self.sim_id_ = sim_id
        self.robot_id_ = robot_id
        if self.use_aj_:
            self.aji_ = np.asarray([self.n2i_(n) for n in self.ajn_])
        if self.use_pj_:
            self.pji_ = np.asarray([self.n2i_(n) for n in self.pjn_])

    def sense(self, out=None):
        sim_id = self.sim_id_
        robot_id = self.robot_id_

        if out is None:
            out = np.zeros(self.size_, dtype=np.float32)
        if self.use_aj_:
            self.sense_joints(
                self.aji_, out=out[self.saj_])
            self.sense_joint_velocities(
                self.aji_, out=out[self.sajv_])
            self.sense_joint_torques(
                self.aji_, out=out[self.sajt_])
        if self.use_pj_:
            self.sense_joints(
                self.pji_, out=out[self.spj_])
            self.sense_joint_velocities(
                self.pji_, out=out[self.spjv_])
            self.sense_joint_torques(
                self.pji_, out=out[self.spjt_])
        if self.use_bp_:
            self.sense_base_position(out=out[self.sbp_])
        if self.use_bv_:
            self.sense_base_velocity(out=out[self.sbv_])
        return out
