#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict
import numpy as np
import pybullet as pb
from typing import List, Callable, Any, Tuple

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

    def __init__(self, sim_id: int = -1, robot_id: int = -1):
        """Initialize the BasePoseSensor

        Args:
            sim_id (int, optional): the pybullet physicsClientId. Defaults to
                -1.
            robot_id (int, optional): the id of the robot to get the pose of. 
                Defaults to -1.
        """
        self.sim_id = sim_id
        self.robot_id = robot_id

    def reset(self, sim_id: int, robot_id: int):
        """Reset the sensor with the new simulation ids

        Args:
            sim_id (int): the new pybullet physicsClientId.
            robot_id (int): the new id of the robot to get the pose of.
        """
        # Save IDs...
        self.sim_id = sim_id
        self.robot_id = robot_id

    def sense(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the position and orientation of the robot

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple comprising
                (np.ndarray): A 1x3 vector representing the position
                (np.ndarray): A 1x4 vector representing the quaternion
                    quaternion orientation (x,y,z,w)
        """
        txn, rxn = pb.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.sim_id)
        return (np.ndarray(txn), np.ndarray(rxn))

    @property
    def sim_id(self) -> int:
        """The sensor physicsClientId

        Returns:
            int: the physicsClientId the sensor is attached to
        """
        return self._sim_id

    @property
    def robot_id(self) -> int:
        """The sensor robot id

        Returns:
            int: the robot id that the sensor is attached to
        """
        return self._robot_id

    @property
    def size(self):
        """The size of the sensor's output space

        Returns:
            int: 7, 3 for position and 4 for orientation
        """
        return 7


class JointStateSensor(SensorBase):
    """
    Pybullet sensor for joint states.
    """

    def __init__(self, joint_indices: List[int] = None,
                 sim_id: int = -1, robot_id: int = -1):
        """Initialize the JointStateSensor

        Args:
            joint_indices (List[int], optional): the joint indices to create
                the sensor for.
            sim_id (int, optional): the pybullet physicsClientId. Defaults to
                -1.
            robot_id (int, optional): the id of the robot to get the pose of. 
                Defaults to -1.
        """
        # handles
        self.sim_id = sim_id
        self.robot_id = robot_id

        # params
        self._joint_indices = joint_indices
        self._get_num_joints()

        # cfg
        self._space = None

    def _get_num_joints(self):
        """Get the number of joints
        """
        # Get relevant parameters.
        if self._joint_indices is None:
            # FIXME(ycho): what if # of joints changes per reset?
            self._num_joints = pb.getNumJoints(
                self.robot_id, physicsClientId=self.sim_id)
            self._joint_indices = list(range(self._num_joints))
        else:
            self._num_joints = len(self._joint_indices)

    def reset(self, sim_id: int, robot_id: int):
        """Reset the sensor with the new simulation ids

        Args:
            sim_id (int): the new pybullet physicsClientId.
            robot_id (int): the new id of the robot to get the pose of.
        """
        # Save IDs...
        self.sim_id = sim_id
        self.robot_id = robot_id

        # Get relevant parameters.
        self._get_num_joints()

        # FIXME(ycho): Determine `space` type based on prismatic/revolute
        jlim_lo = np.zeros(self._num_joints)
        jlim_hi = np.zeros(self._num_joints)
        for i, j in enumerate(self._joint_indices):
            joint_info = pb.getJointInfo(robot_id, j, physicsClientId=sim_id)
            joint_type = joint_info[2]

            lo = joint_info[8]
            hi = joint_info[9]

            jlim_lo[i] = lo
            jlim_hi[i] = hi

        self._space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def sense(self) -> List[Any]:
        """Get the joint states

        Returns:
            List[Any]: A list of length len(self._joint_indices), where each
                element in the list contains 4 values which correspond to:
                    (float): The position value of this joint.
                    (float): The velocity value of this joint.
                    (List[float]): The joint reaction forces, if a torque sensor is
                        enabled forthis joint it is [Fx, Fy, Fz, Mx, My, Mz]. 
                        Without torque sensor, it is [0,0,0,0,0,0].
                    (float): This is the motor torque applied during the last
                        stepSimulation.Note that this only applies in
                        VELOCITY_CONTROL and POSITION_CONTROL. If you use
                        TORQUE_CONTROL then theapplied joint motor torque is
                        exactly what you provide, so there isno need to report
                        it separately.
        """
        joint_states = pb.getJointStates(
            self.robot_id, self._joint_indices, physicsClientId=self.sim_id)
        return joint_states

    @property
    def sim_id(self):
        """The sensor physicsClientId

        Returns:
            int: the physicsClientId the sensor is attached to
        """
        return self._sim_id

    @property
    def robot_id(self):
        """The sensor robot id

        Returns:
            int: the robot id that the sensor is attached to
        """
        return self._robot_id

    @property
    def space(self):
        """The size of the sensor's output space

        Returns:
            int: equal to the number of joints either given by joint indices
                or the number of joints present on the robot if joint indices
                was not provided.
        """
        return self._space


class FlatSensorWrapper(SensorBase):
    """A wrapper for an OrderedDict of sensors which provides a flattened
    sense and space interface to act as a single sensor.
    """

    def __init__(self, sensors: OrderedDict):
        """Initialize a FlatSensorWrapper

        Args:
            sensors (OrderedDict): An ordered dict whose keys are strings and
                values are SensorBase objects.
        """
        assert isinstance(sensors, OrderedDict), 'sensors must be OrderedDict'
        self._sensors = sensors
        self._structure = OrderedDict(
            [(k, s.space) for k, s in sensors.items()])
        self._space = gym.spaces.flatten_space(self._structure)

    @property
    def space(self) -> gym.Space:
        """The combined sensor's output space which is the concatenation of
        the sensor's spaces from the input sensors.

        Returns:
            (gym.Space): The combined output space
        """
        return self._space

    @property
    def sim_id(self):
        """The sensor physicsClientId

        Returns:
            int: the physicsClientId the sensor is attached to
        """
        return self._sim_id

    @property
    def robot_id(self):
        """The sensor robot id

        Returns:
            int: the robot id that the sensors are attached to
        """
        return self._robot_id

    def sense(self) -> np.ndarray:
        """Returns a flattened 1d array 

        Returns:
            np.ndarray: A 1d numpy array which is the concatenated sensor
                values.
        """
        return gym.spaces.flatten(
            self._structure, {k: s.sense() for k, s in self._sensors})


class PybulletPhonebotSensor():
    def __init__(self,
                 sense_active_joints: bool = True,
                 sense_passive_joints: bool = True,
                 sense_base_position: bool = True,
                 sense_base_velocity: bool = True,
                 active_joint_names: List[str] = None,
                 passive_joint_names: List[str] = None,
                 index_from_name: Callable[[str], int] = None
                 ):
        """Initialize PybulletPhonebotSensor

        Args:
            sense_active_joints (bool, optional): If True, then sense the active
                joints of Phonebot. Defaults to True.
            sense_passive_joints (bool, optional): If True, then sense the
                passive joints of Phonebot. Defaults to True.
            sense_base_position (bool, optional): If True, then sense the base
                position of Phonebot, which consists of x, y, z, qx, qy, qz, qw.
                "xyz" is the position, and "q_" is a quaternion representing
                orientation.
                Defaults to True.
            sense_base_velocity (bool, optional): If True, then sense the base
                velocity of Phonebot, which consists of linear and angular
                velocities, vx vy vz and wx wy wz. Defaults to True.
            active_joint_names (List[str], optional): A list of names of
                the active joints. This is required if sense_active_joints
                is True. Defaults to None.
            passive_joint_names (List[str], optional): A list of names of
                the passive joints. This is required if sense_passive_joints
                is True. Defaults to None.
            index_from_name (Callable[[str], int], optional): If either of
                sense_active_joints or sense_passive_joints is True, then this
                must be provided. This is a callable which converts a joint
                name to an index. Defaults to None.

        Raises:
            ValueError: Active joint indices must be supplied for extracting
                joint states.
            ValueError: Passive joint indices must be supplied for extracting
                joint states.
        """

        # Configure sensing state.
        self._sense_aj = sense_active_joints
        self._sense_pj = sense_passive_joints
        self._sense_bp = sense_base_position
        self._sense_bv = sense_base_velocity

        # Save joint indices.
        self._ajn = active_joint_names
        self._pjn = passive_joint_names
        self._n2i = index_from_name
        self._aji = None
        self._pji = None

        if (self._sense_aj and self._ajn is None):
            raise ValueError(
                "Joint indices must be supplied for extracting joint states.")

        if (self._sense_pj and self._pjn is None):
            raise ValueError(
                "Joint indices must be supplied for extracting joint states.")

        # Compute state size.
        state_config = self.compute_state_config()
        (self._saj, self._sajv, self._sajt, self._spj,
            self._spjv, self._spjt, self._sbp, self._sbv,
            self._size) = state_config

        # Compute observation space ...
        lo = np.zeros(self._size, dtype=np.float32)
        hi = np.zeros(self._size, dtype=np.float32)

        # active joint limits
        if self._sense_aj:
            lo[self._saj] = -np.pi / 2
            hi[self._saj] = +np.pi / 2
            lo[self._sajv] = -np.inf
            hi[self._sajv] = +np.inf
            lo[self._sajt] = -np.inf
            hi[self._sajt] = +np.inf

        if self._sense_pj:
            lo[self._spj] = -np.pi
            hi[self._spj] = +np.pi
            lo[self._spjv] = -np.inf
            hi[self._spjv] = +np.inf
            lo[self._spjt] = -np.inf
            hi[self._spjt] = +np.inf

        if self._sense_bp:
            lo[self._sbp] = -np.inf
            hi[self._sbp] = +np.inf

        if self._sense_bv:
            lo[self._sbv] = -np.inf
            hi[self._sbv] = +np.inf

        self._space = gym.spaces.Box(
            lo, hi, shape=(self._size,), dtype=np.float32)

    @property
    def size(self) -> int:
        """The size of the output space

        Returns:
            int: The number of sensed values returned
        """
        return self._size

    @property
    def slice_active_joints(self) -> slice:
        """A slice to get the active joints from the sensor

        Returns:
            slice: The slice into the sense array to return the active joint
                position values
        """
        return self._saj

    @property
    def slice_passive_joints(self) -> slice:
        """A slice to get the passive joints from the sensor

        Returns:
            slice: The slice into the sense array to return the passive joint
                position values
        """
        return self._spj

    @property
    def slice_active_joints_velocity(self) -> slice:
        """A slice to get the active joint velocities from the sensor

        Returns:
            slice: The slice into the sense array to return the active joint
                velocity values
        """
        return self._sajv

    @property
    def slice_passive_joints_velocity(self) -> slice:
        """A slice to get the passive joint velocities from the sensor

        Returns:
            slice: The slice into the sense array to return the passive joint
                velocity values
        """
        return self._spjv

    @property
    def slice_active_joints_torque(self) -> slice:
        """A slice to get the active joint torques from the sensor

        Returns:
            slice: The slice into the sense array to return the active joint
                torques values
        """
        return self._sajt

    @property
    def slice_passive_joints_torque(self) -> slice:
        """A slice to get the passive joint torques from the sensor

        Returns:
            slice: The slice into the sense array to return the passive joint
                torques values
        """
        return self._spjt

    @property
    def slice_base_position(self) -> slice:
        """A slice to get the base position from the sensor

        Returns:
            slice: The slice into the sense array to return the position values
        """
        return self._sbp

    @property
    def slice_base_velocity(self) -> slice:
        """A slice to get the base velocity from the sensor

        Returns:
            slice: The slice into the sense array to return the velocity values
        """
        return self._sbv

    @property
    def space(self) -> gym.Space:
        """The sense space object

        Returns:
            gym.Space: The sense space object the size of the number of readings
                sensed.
        """
        return self._space

    def compute_state_config(self) -> Tuple[slice, slice, slice, slice, slice, slice, slice, slice, int]:
        """Compute the sensor values to read and return the appropriate
        slices to access the sense readings.

        Returns:
            Tuple[slice, slice, slice, slice, slice, slice, slice, slice, int]: 
                A tuple comprising:
                    (slice): Active joint positions slice
                    (slice): Active joint velocities slice
                    (slice): Active joint torques slice
                    (slice): Passive joint positions slice
                    (slice): Passive joint velocities slice
                    (slice): Passive joint torques slice
                    (slice): Base position slice (x, y, z, qx, qy, qz, qw)
                    (slice): Base velocity slice (vx, vy, vz, rx, ry, rz)
                    (int): The number of sensor readings to gather
        """
        n = 0
        # Active joint pos, vel, torque
        dn = self._sense_aj * 8
        saj = np.s_[n:n + dn]
        n += dn

        dn = self._sense_aj * 8
        sajv = np.s_[n:n + dn]
        n += dn

        dn = self._sense_aj * 8
        sajt = np.s_[n:n + dn]
        n += dn

        # Passive joint pos, vel, torque
        dn = self._sense_pj * 8
        spj = np.s_[n:n + dn]
        n += dn

        dn = self._sense_pj * 8
        spjv = np.s_[n:n + dn]
        n += dn

        dn = self._sense_pj * 8
        spjt = np.s_[n:n + dn]
        n += dn

        dn = self._sense_bp * 7
        sbp = np.s_[n:n + dn]
        n += dn

        dn = self._sense_bv * 6
        sbv = np.s_[n:n + dn]
        n += dn

        return (saj, sajv, sajt, spj, spjv, spjt, sbp, sbv, n)

    def sense_base_position(self, out: np.ndarray = None) -> np.ndarray:
        """Sense the base position of the Phonebot

        Args:
            out (np.ndarray, optional): If provided, then update the data in the
                provided 1d vector with the result as well as return the
                sensed position. Defaults to None.

        Returns:
            np.ndarray: The output array (1x7 vector) of the x, y, z, qx, qy,
                qz, qw position and orientation.
        """
        sim_id = self._sim_id
        robot_id = self._robot_id
        if out is None:
            out = np.zeros(7, dtype=np.float32)
        bx, bq = pb.getBasePositionAndOrientation(
            robot_id, physicsClientId=sim_id)
        out[0:3] = bx
        out[3:7] = bq
        return out

    def sense_base_velocity(self, out: np.ndarray = None) -> np.ndarray:
        """Sense the base velocity of the Phonebot

        Args:
            out (np.ndarray, optional): If provided, then update the data in the
                provided 1d vector with the result as well as return the
                sensed velocity. Defaults to None.

        Returns:
            np.ndarray: The output array (1x6 vector) of the vx, vy, vz, rx, ry,
                rz linear and angular velocities.
        """
        sim_id = self._sim_id
        robot_id = self._robot_id
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
        sim_id = self._sim_id
        robot_id = self._robot_id
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
        sim_id = self._sim_id
        robot_id = self._robot_id
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
        sim_id = self._sim_id
        robot_id = self._robot_id
        shape = np.shape(joint_indices)
        if out is None:
            out = np.zeros(shape, dtype=np.float32)
        joint_states = pb.getJointStates(
            robot_id, np.ravel(joint_indices), physicsClientId=sim_id)

        jtorque = [j[3] for j in joint_states]
        out[...] = np.reshape(jtorque, out.shape)
        return out

    def reset(self, sim_id: int, robot_id: int):
        """Reset the simulation with the new robot and sim id

        Args:
            sim_id (int, optional): the pybullet physicsClientId.
            robot_id (int, optional): the id of the robot to sense.
        """
        self._sim_id = sim_id
        self._robot_id = robot_id
        if self._sense_aj:
            self._aji = np.asarray([self._n2i(n) for n in self._ajn])
        if self._sense_pj:
            self._pji = np.asarray([self._n2i(n) for n in self._pjn])

    def sense(self, out: np.ndarray = None) -> np.ndarray:
        """Sense the values requested upon initialization.

        Args:
            out (np.ndarray, optional): If provided, then update the data in the
                provided 1d vector with the result as well as return the
                sensed reading. The array must be at least of length #N where
                N is the size of the sense space. Defaults to None.

        Returns:
            np.ndarray: An #Nx1 vector of sensed values which
                is defined based on instance initialization
        """
        if out is None:
            out = np.zeros(self._size, dtype=np.float32)
        if self._sense_aj:
            self.sense_joints(
                self._aji, out=out[self._saj])
            self.sense_joint_velocities(
                self._aji, out=out[self._sajv])
            self.sense_joint_torques(
                self._aji, out=out[self._sajt])
        if self._sense_pj:
            self.sense_joints(
                self._pji, out=out[self._spj])
            self.sense_joint_velocities(
                self._pji, out=out[self._spjv])
            self.sense_joint_torques(
                self._pji, out=out[self._spjt])
        if self._sense_bp:
            self.sense_base_position(out=out[self._sbp])
        if self._sense_bv:
            self.sense_base_velocity(out=out[self._sbv])
        return out
