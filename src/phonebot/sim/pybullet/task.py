#!/usr/bin/env python3

from abc import abstractmethod, ABC, abstractproperty

import numpy as np
import pybullet as pb
from typing import List, Tuple
from dataclasses import dataclass
import gym

from phonebot.sim.pybullet.sensor import PybulletPhonebotSensor
from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.math.utils import anorm

logger = get_default_logger()


class BaseTask(ABC):

    @staticmethod
    @abstractmethod
    def get_size():
        pass

    @abstractmethod
    def reset(self, sim_id: int, robot_id: int, state: np.ndarray):
        pass

    @abstractmethod
    def compute_reward(self, state0: np.ndarray, state1: np.ndarray):
        pass

    @abstractmethod
    def compute_is_done(self, state: np.ndarray):
        pass

    @abstractproperty
    def space(self):
        pass





@dataclass
class ForwardVelocityTaskSettings:
    random_init: bool = True
    init_ang_scale: Tuple[float, float, float] = (
        np.pi / 4, np.pi / 4, np.pi / 4)


class ForwardVelocityTask(BaseTask):
    space = gym.spaces.Box(0, 0, shape=(0,))

    def __init__(self, sensor: PybulletPhonebotSensor,
                 settings: ForwardVelocityTaskSettings):
        self.sensor_ = sensor
        self.settings = settings
        self.fwd_ = np.zeros(3, dtype=np.float32)
        self.rng = np.random

    @staticmethod
    def get_size():
        return 0

    @property
    def target(self):
        return []

    def set_rng(self, rng):
        self.rng = rng

    def reset(self, sim_id: int, robot_id: int):
        # Reset robot based on settings.
        position = [0, 0, 0]
        rotation = [0, 0, 0, 1]
        if self.settings.random_init:
            ang_scale = np.asarray(self.settings.init_ang_scale)
            rot_euler = self.rng.uniform(-ang_scale, ang_scale)
            rotation = pb.getQuaternionFromEuler(rot_euler)
        pb.resetBasePositionAndOrientation(
            robot_id, position, rotation, physicsClientId=sim_id)

        # Determine forward direction from now...
        pos, rot = pb.getBasePositionAndOrientation(
            robot_id, physicsClientId=sim_id)
        yaw = pb.getEulerFromQuaternion(rot)[2]
        self.fwd_ = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])

    def compute_reward(self, state0, state1):
        # vel0 = state0[self.sensor_.slice_base_velocity]
        vel1 = state1[self.sensor_.slice_base_velocity]
        #dv = vel1[0:3] - vel0[0:3]
        # return np.dot(self.fwd_, dv)
        return np.dot(self.fwd_, vel1[0:3])

    def compute_is_done(self, state):
        return False


@dataclass
class CounterClockwiseRotationTaskSettings:
    random_init: bool = True
    init_ang_scale: Tuple[float, float, float] = (
        np.pi / 4, np.pi / 4, np.pi / 4)


class CounterClockwiseRotationTask(BaseTask):
    space = gym.spaces.Box(0, 0, shape=(0,))

    def __init__(self, sensor: PybulletPhonebotSensor,
                 settings: CounterClockwiseRotationTaskSettings):
        self.sensor_ = sensor
        self.settings = settings
        self.rng = np.random

    @staticmethod
    def get_size():
        return 0

    @property
    def target(self):
        return []

    def set_rng(self, rng):
        self.rng = rng

    def reset(self, sim_id: int, robot_id: int):
        # Reset robot based on settings.
        position = [0, 0, 0]
        rotation = [0, 0, 0, 1]
        if self.settings.random_init:
            ang_scale = np.asarray(self.settings.init_ang_scale)
            rot_euler = self.rng.uniform(-ang_scale, ang_scale)
            rotation = pb.getQuaternionFromEuler(rot_euler)
        pb.resetBasePositionAndOrientation(
            robot_id, position, rotation, physicsClientId=sim_id)

    def compute_reward(self, state0, state1):
        vel1 = state1[self.sensor_.slice_base_velocity]
        return vel1[5]

    def compute_is_done(self, state):
        return False


class HoldVelocityTask(BaseTask):
    space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))

    def __init__(self, sensor: PybulletPhonebotSensor):
        self.sensor_ = sensor
        self.target_v = 0.0
        self.target_w = 0.0
        self.weight_v = 1.0
        self.weight_w = 0.1
        self.rng = np.random

    @staticmethod
    def get_size():
        return 2

    @property
    def target(self):
        return np.asarray([self.target_v, self.target_w])

    def set_rng(self, rng):
        self.rng = rng

    def reset(self, sim_id: int, robot_id: int):
        # TODO(ycho): use seeded contextual RNG rather than np.random directly.
        self.target_v = self.rng.uniform(0.01, 0.5)
        self.target_w = self.rng.uniform(-0.1, 0.5)

    def compute_reward(self, state0, state1):
        bp = state1[self.sensor_.slice_base_position]
        x = bp[0:3]
        q = bp[3:7]

        bv = state1[self.sensor_.slice_base_velocity]
        v = bv[0:3]
        w = bv[3:6]

        # Convert to base frame.
        qi = [q[0], q[1], q[2], -q[3]]  # NOTE(ycho) qi=inverse
        v_x = pb.rotateVector(qi, v)[0]
        w_z = pb.getEulerFromQuaternion(q)[2]

        err_v = (self.target_v - v_x)
        err_w = (self.target_w - w_z)
        #logger.debug(
        #    '{:.2f} {:.2f} {:.2f} {:.2f}'.format(
        #        v_x, w_z, self.target_v, self.target_w)
        #)

        err = (self.weight_v * err_v * err_v +
               self.weight_w * err_w * err_w)

        return -err

    def compute_is_done(self, state):
        return False



class HoldSpeedTask(BaseTask):
    space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
    
    def __init__(self, sensor: PybulletPhonebotSensor):
        self.sensor_ = sensor
        self.target_speed = 0.0
        self.rng = np.random
        
    @staticmethod
    def get_size():
        return 1
    
    @property
    def target(self):
        return self.target_speed

    def set_rng(self, rng):
        self.rng = rng

    def reset(self, sim_id: int, robot_id: int):
        self.target_speed = self.rng.uniform(0.01, 1)

    def compute_reward(self, state0: np.ndarray, state1: np.ndarray):
        """Computes the reward by taking the difference between the current
        speed and the target speed, negating and squaring it. The Z component
        of velocity is removed before the speed is calculated.

        R = -(s_targ - s_curr)**2

        Returns:
            float: Reward calculated as described above.
        """

        bv = state1[self.sensor_.slice_base_velocity]
        v = bv[0:3]
        v[2] = 0 
        err_v = (self.target_speed - np.linalg.norm(v))**2

        return -err_v

    def compute_is_done(self, state: np.ndarray):
        return False


    def calc_state_energy(self, prev_state: np.ndarray, curr_state: np.ndarray)-> float:
        """Calculates the amount of energy expended going from the previous state
        to the current state. Specifically, use a linear estimate of the work
        done to get from the previous state to the current state assuming
        that the torque for each motor is the average torque from prev_state
        to curr_state. 

        This is an estimate of the change in state energy, and is not exact, and
        may result in drift over time if used with conservation laws.

        Returns:
            float: Energy (total work) required to move from previous state to
                current state.
        """
        joint_pos_1 = state1[self.sensor_.slice_active_joints]
        joint_pos_2 = state2[self.sensor_.slice_active_joints]
        joint_d_pos = joint_pos_2 - joint_pos_1
        joint_torque_1 = state1[self.sensor_.slice_active_joints_torque]
        joint_torque_2 = state2[self.sensor_.slice_active_joints_torque]
        mean_torque = (joint_torque_1 + joint_torque_2)/2

        work = joint_d_pos*mean_torque
        total_work = np.sum(work)
        return total_work


@dataclass
class ReachPositionTaskSettings:
    max_time: float

    # Whether to initialize randomly
    random_init: bool = False

    # Uniform distribution about zero.
    init_ang_scale: Tuple[float, float, float] = (
        np.deg2rad(0), np.deg2rad(0), np.pi)

    v_max: float = 0.1
    w_max: float = 0.0


class ReachPositionTask(BaseTask):
    space = gym.spaces.Box(
        np.array([-np.inf, -np.inf, -np.pi]),
        np.array([np.inf, np.inf, np.pi]),
        shape=(3,),
        dtype=np.float32)

    def __init__(self, settings: ReachPositionTaskSettings,
                 sensor: PybulletPhonebotSensor):
        self.settings = settings
        self.sensor_ = sensor

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_h = 0.0

        self.max_err_txn = 0.04
        self.max_err_rxn = np.deg2rad(20.0)
        self.max_err = np.asarray(
            [self.max_err_txn, self.max_err_txn, self.max_err_rxn])

    @staticmethod
    def get_size():
        return 3

    @property
    def source(self):
        return np.asarray([0, 0, 0])

    @property
    def target(self):
        return np.asarray([self.target_x, self.target_y, self.target_h])

    @property
    def weight(self):
        return np.asarray([1.0, 1.0, 0.1])

    @staticmethod
    def _generate_target_position(v_max: float, w_max: float, max_time: float):
        # TODO(ycho): use seeded contextual RNG rather than np.random directly.
        dt = max_time * np.random.uniform(0.3, 0.7)
        v = np.random.uniform(0.5 * v_max, v_max)
        w = np.random.uniform(-w_max, w_max)
        r = v / (w + np.finfo(np.float32).eps)
        h = w * dt
        # Deal with numerical instabilities around small ang vel.
        if np.abs(h) < np.finfo(np.float32).eps:
            x = v * dt
            y = 0.0
        else:
            x = r * np.sin(h)
            y = r - np.cos(h) * r

        return (x, y, h)

    def reset(self, sim_id: int, robot_id: int):
        # Reset robot based on settings.
        position = [0, 0, 0]
        rotation = [0, 0, 0, 1]
        if self.settings.random_init:
            ang_scale = np.asarray(self.settings.init_ang_scale)
            rot_euler = np.random.uniform(-ang_scale, ang_scale)
            rotation = pb.getQuaternionFromEuler(rot_euler)
        pb.resetBasePositionAndOrientation(
            robot_id, position, rotation, physicsClientId=sim_id)

        # Compute relative target from current position ...
        yaw = pb.getEulerFromQuaternion(rotation)[2]
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy, dh = self._generate_target_position(
            self.settings.v_max, self.settings.w_max, self.settings.max_time)

        # Set targets computed as offsets from the current state.
        self.target_h = anorm(yaw + dh)
        self.target_x = position[0] + c * dx - s * dy
        self.target_y = position[1] + s * dx + c * dy

        # Cache max cost term for reward computation.
        delta = self._compute_error(self.target, self.source)
        self.max_cost_ = np.linalg.norm(self.weight * delta)

    def _encode_state_2d(self, state):
        """ encode state from SE3 -> SE2 """
        s = state[self.sensor_.slice_base_position]
        x = s[0:3]
        q = s[3:7]
        yaw = pb.getEulerFromQuaternion(q)[2]
        return np.asarray([x[0], x[1], yaw])

    @staticmethod
    def _compute_error(s0, s1):
        err = s0 - s1
        err[2] = anorm(err[2])
        return err

    def compute_reward(self, state0, state1):
        # Simplify state terms and compute error wrt target.
        s0 = self._encode_state_2d(state0)
        s1 = self._encode_state_2d(state1)
        d0 = self._compute_error(s0, self.target)
        d1 = self._compute_error(s1, self.target)

        # Reward = normalized relative cost improvement
        cost0 = np.linalg.norm(self.weight * d0)
        cost1 = np.linalg.norm(self.weight * d1)
        reward = -(cost1 - cost0) / self.max_cost_

        done = self.compute_is_done(state1)
        return 1.0 if done else reward

    def compute_is_done(self, state):
        # Parse state.
        bp = state[self.sensor_.slice_base_position]
        bx = bp[0:3]
        bq = bp[3:7]

        # Compute error.
        dx = self.target_x - bx[0]
        dy = self.target_y - bx[1]
        dh = anorm(self.target_h - pb.getEulerFromQuaternion(bq)[2])
        return np.less(np.abs([dx, dy, dh]), self.max_err).all()
