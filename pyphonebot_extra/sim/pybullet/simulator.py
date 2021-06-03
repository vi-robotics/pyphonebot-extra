#!/usr/bin/env python3

from typing import Any, List, Tuple, Dict
import time
import numpy as np
import pybullet as pb
import math
from dataclasses import dataclass
from typing import List, Tuple

import gym
import gym.utils.seeding as seeding
from gym.envs.registration import register

from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.utils import anorm
from phonebot.core.controls.pid import PID, PIDSettings
from pyphonebot_extra.sim.common.phonebot_model import PhonebotModel
from pyphonebot_extra.sim.pybullet.builder import PybulletBuilder
from pyphonebot_extra.sim.pybullet.debug_utils import (
    debug_draw_inertia_box, debug_draw_frame_axes, debug_get_full_aabb
)
from pyphonebot_extra.sim.pybullet.sensor import PybulletPhonebotSensor
from pyphonebot_extra.sim.pybullet.task import (
    ForwardVelocityTask, ForwardVelocityTaskSettings,
    HoldVelocityTask,
    ReachPositionTask, ReachPositionTaskSettings,
    CounterClockwiseRotationTask, CounterClockwiseRotationTaskSettings
)

from phonebot.core.common.logger import get_default_logger
logger = get_default_logger()


@dataclass
class PybulletSimulatorSettings(Settings):

    debug: bool = False
    debug_axis_scale: float = 0.02
    debug_inertia: bool = True
    debug_midair: bool = False
    debug_contact: bool = False
    debug_follow_camera: bool = True
    debug_solver: bool = False
    debug_servo: bool = False
    debug_show_trajectory: bool = True

    # Physics Settings
    gravity: float = -9.81
    # NOTE(yycho0108): chosen somewhat arbitrarily.
    # self.timestep = (1.0 / 240)  # NOTE(yycho0108): pybullet default
    timestep: float = 0.008
    max_solver_iter: int = 4096
    solver_residual_threshold: float = 1e-6
    realtime: bool = False
    timestep_range: Tuple[float, float] = (0.001, 0.016)

    # Render settings
    render: bool = True
    # TODO(yycho0108): Ugly parameter... derive this from phonebot dimensions.
    record_file: str = ''
    show_pybullet_gui: bool = False

    # Servo motor parameters.
    # TODO(ycho): Currrent servo motor based on PID
    # Does not work (esp. wrt certain timestep settings).
    # Maybe a higher fidelity motor model would work better,
    # but for now recommended to disable by default.
    use_torque_control: bool = False
    servo_kp: float = 0.02
    servo_ki: float = 1.0
    servo_kd: float = 0.002
    joint_friction_force: float = 0.0
    clip_torque: bool = False
    clip_velocity: bool = False

    # Friction parameters.
    # TODO(ycho): Figure out if these params are reasonable.
    lateral_friction: float = 0.99
    spinning_friction: float = 0.01
    rolling_friction: float = 0.01

    # Other environment specific parameters (config)
    max_num_steps: int = 1000
    zero_at_nominal: bool = True
    normalize_action_space: bool = True
    random_timestep: bool = False
    start: bool = True

    def __post_init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def description(self):
        return self.__doc__


class PybulletPhonebotEnv(gym.Env):
    """
    Pybullet Simulator.
    NOTE(yycho0108): Unlike PybulletBuilder,
    this class not intended for general-purpose modelling;
    i.e. only configured to run Phonebot.
    """
    metadata = {
        'render.modes': ['none', 'human']
    }

    # TODO(ycho): action_space should depend on the controller of the environment.
    # i.e. ranging from raw joint angle controller to endpoint controller, etc.
    # action_space = gym.spaces.Box(-np.pi/2, np.pi/2, shape=(8,))

    def __init__(self,
                 sim_settings: PybulletSimulatorSettings = None,
                 phonebot_settings: PhonebotSettings = None
                 ):
        """A PybulletPhonebotEnv gym environment

        Args:
            sim_settings (PybulletSimulatorSettings, optional): Settings
                for the Pybullet simulator. Defaults to None.
            phonebot_settings (PhonebotSettings, optional): Settings to
                describe PhoneBot dimensions. Defaults to None.
        """
        super().__init__()
        if sim_settings is None:
            sim_settings = PybulletSimulatorSettings()
        if phonebot_settings is None:
            phonebot_settings = PhonebotSettings()
        self.settings = sim_settings
        self.config = phonebot_settings
        self.model = PhonebotModel(self.config)
        self.builder = self.build(self.model)
        self.np_random = np.random

        # hmm...
        self.servo_torque = self.config.servo_torque

        # Initialize runtime variables.
        self.sim_id = -1
        self.phonebot_id = -1
        s = self.settings
        # FIXME(ycho): max_i set to something that feels reasonable ...
        # hardcoded.
        self.pid_ = PID(
            PIDSettings(
                s.servo_kp, s.servo_ki, s.servo_kd,
                max_i=(4 * self.servo_torque / s.servo_ki
                       if s.servo_ki > 0 else 0),
                max_u=self.servo_torque, cutoff_freq=1.0 /
                (16.0 * self.settings.timestep + 1e-8)))
        self.time_index = 0
        self.time = 0.0

        # Sensor + task
        self._sensor = PybulletPhonebotSensor(True, True, True, True,
                                              self.config.active_joint_names,
                                              self.config.passive_joint_names,
                                              self._index_from_joint)

        # Change this task and settings to change the RL agent target and
        # environment initialization
        self.task_ = CounterClockwiseRotationTask(
            self._sensor, CounterClockwiseRotationTaskSettings())
        self.task_.set_rng(self.np_random)

        if sim_settings.normalize_action_space:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,))
        else:
            self.action_space = gym.spaces.Box(-np.pi / 2,
                                               np.pi / 2, shape=(8,))

        # Configure observation space as a combination of sensor+task.
        # FIXME(ycho): NOT a generalized construct, specific to
        # our current sensor + task combo.
        s_lo = self._sensor.space.low
        s_hi = self._sensor.space.high
        t_lo = self.task_.space.low
        t_hi = self.task_.space.high
        lo = np.r_[s_lo, t_lo]
        hi = np.r_[s_hi, t_hi]
        self.observation_space = gym.spaces.Box(lo, hi)

        self.task_debug_ids_ = []
        # Initialize the global engine immediately.
        if self.settings.start:
            self.start()

    @ property
    def sensor(self) -> PybulletPhonebotSensor:
        """The PhoneBot sensor

        Returns:
            PybulletPhonebotSensor: the environment PhoneBot sensor
        """
        return self._sensor

    def build(self, model: PhonebotModel) -> PybulletBuilder:
        """Build the PhoneBot model

        Args:
            model (PhonebotModel): A given PhonebotModel

        Returns:
            PybulletBuilder: The finalized builder
        """
        joints, links = model.build()
        builder = PybulletBuilder()
        for joint in joints:
            builder.add_joint(joint)
        for link in links:
            builder.add_link(link)
        builder.finalize()
        return builder

    def _index_from_joint(self, joint: str) -> int:
        return self.builder.index_from_joint_[joint]

    def _index_from_link(self, link: str) -> int:
        return self.builder.index_from_link_[link]

    def _joint_from_index(self, index: int) -> str:
        return self.builder.joint_from_index_[index]

    def _link_from_index(self, index: int) -> str:
        return self.builder.link_from_index_[index]

    def start(self):
        """Start the simulator
        """
        # Unroll parameters.
        builder = self.builder
        config = self.config

        # Create and configure simulation.
        if self.settings.render:
            sim_id = pb.connect(pb.GUI)
        else:
            sim_id = pb.connect(pb.DIRECT)
        self.sim_id = sim_id
        pb.setGravity(0, 0, self.settings.gravity,
                      physicsClientId=sim_id)
        pb.setTimeStep(self.settings.timestep, physicsClientId=sim_id)
        pb.setRealTimeSimulation(
            self.settings.realtime, physicsClientId=sim_id)
        # TODO(ycho): Magic constants everywhere ...
        pb.setPhysicsEngineParameter(
            numSolverIterations=self.settings.max_solver_iter,
            # numSubSteps=1,
            # erp=1.0,
            solverResidualThreshold=self.settings.solver_residual_threshold,
            reportSolverAnalytics=self.settings.debug_solver,
            physicsClientId=sim_id
        )

        # Add ground plane @ z=0.
        if True:
            plane = pb.createCollisionShape(
                pb.GEOM_PLANE, physicsClientId=sim_id)
            plane_index = pb.createMultiBody(0, plane, physicsClientId=sim_id)
            if True:
                # TODO(ycho): Evaluate if setting friction params is necessary.
                pb.changeDynamics(
                    plane_index, -1,
                    lateralFriction=self.settings.lateral_friction,
                    spinningFriction=self.settings.spinning_friction,
                    rollingFriction=self.settings.rolling_friction,
                    physicsClientId=sim_id)
            pb.resetBasePositionAndOrientation(
                plane_index, [0, 0, 0.0], [0, 0, 0, 1], physicsClientId=sim_id)
            self.plane_id = plane_index

        # Add phonebot.
        phonebot_id = builder.create(sim_id)
        self.phonebot_id = phonebot_id

        # Post-Configure env.
        # FIXME(ycho): Should this be a mixin instead?
        # i.e. class
        # PybulletPhonebotFullStateForwardVelocityEnv(PybulletPhonebotEnv,
        # ActiveJointsController, FullStateSensor, HoldVelocityTask)

        # Add endpoint connection constraint, which is specific to phonebot.
        # NOTE(yycho0108): `maxForce` is currently set somewhat arbitrarily
        # Such that it could compensate for the joint torque while
        # still maintaining a reasonable degree of connectivity between the two
        # points.

        # NOTE(ycho): Limit max constraint by the ~max exertible
        # force by the servos.
        servo_force = self.servo_torque / config.hip_link_length
        constraint_force = 2 * servo_force
        for prefix in config.order:
            ia = builder.index_from_link_[
                '{}_foot_link_a'.format(prefix)]
            ib = builder.index_from_link_[
                '{}_foot_link_b'.format(prefix)]
            c = pb.createConstraint(
                phonebot_id, ia, phonebot_id, ib, pb.JOINT_POINT2POINT, [
                    0.0, 0.0, 1.0], [
                    0.0, 0.0, 0.0], [
                    0.0, 0.0, 0.0], physicsClientId=sim_id)
            # pb.changeConstraint(
            #    c, maxForce=constraint_force,
            #    physicsClientId=sim_id)

            # TODO(ycho): Figure out if it's actually meaningful
            # to make the constraint "symmetric".
            # c = pb.createConstraint(phonebot_id, ib, phonebot_id, ia, pb.JOINT_POINT2POINT, [
            #     0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=sim_id)
            # pb.changeConstraint(
            #     c, maxForce=constraint_force,
            #     physicsClientId=sim_id)

        # Optionally apply constraint on max joint velocity.
        if self.settings.clip_velocity:
            # FIXME(ycho): Remove hardcoded param
            # 10.47 == 60deg/0.1sec
            for joint in config.active_joint_names:
                index = builder.index_from_joint_[joint]
                pb.changeDynamics(phonebot_id, index,
                                  maxJointVelocity=10.47,
                                  physicsClientId=sim_id)

        # Optionally apply joint limits.
        if True:
            # FIXME(ycho): Pedantically speaking, the joint limits
            # should indeed be from -np.pi/2~np.pi/2 (i.e. 0 @ folded position).
            # However, in order to satisfy the RL stability conditions
            # i.e. symmetric action space normalized in [-1,+1],
            # this is a work-around that slightly violates real joint
            # constraints.
            zero = self.config.nominal_hip_angle if self.settings.zero_at_nominal else 0.0
            for joint in config.active_joint_names:
                index = builder.index_from_joint_[joint]
                pb.changeDynamics(phonebot_id, index,
                                  jointLowerLimit=zero - np.pi / 2,
                                  jointUpperLimit=zero + np.pi / 2,
                                  physicsClientId=sim_id)

        # Optionally enable collision against leg stopper.
        if True:
            for prefix in config.order:
                for suffix in 'ab':
                    pb.setCollisionFilterPair(
                        phonebot_id, phonebot_id,
                        builder.index_from_link_[
                            '{}_hip_link_{}'.format(prefix, suffix)],
                        builder.index_from_link_[
                            '{}_knee_link_{}'.format(prefix, suffix)],
                        0, physicsClientId=sim_id)

                pb.setCollisionFilterPair(
                    phonebot_id, phonebot_id,
                    builder.index_from_link_['{}_foot_link_a'.format(prefix)],
                    builder.index_from_link_['{}_foot_link_b'.format(prefix)],
                    0, physicsClientId=sim_id)

                pb.setCollisionFilterPair(
                    phonebot_id, phonebot_id,
                    builder.index_from_link_['{}_knee_link_a'.format(prefix)],
                    builder.index_from_link_['{}_knee_link_b'.format(prefix)],
                    0, physicsClientId=sim_id)

                pb.setCollisionFilterPair(
                    phonebot_id, phonebot_id,
                    builder.index_from_link_['{}_knee_link_a'.format(prefix)],
                    builder.index_from_link_['{}_foot_link_b'.format(prefix)],
                    0, physicsClientId=sim_id)

                pb.setCollisionFilterPair(
                    phonebot_id, phonebot_id,
                    builder.index_from_link_['{}_knee_link_b'.format(prefix)],
                    builder.index_from_link_['{}_foot_link_a'.format(prefix)],
                    0, physicsClientId=sim_id)

                for suffix in 'ab':
                    for s in ['{}_foot_link_{}', '{}_knee_link_{}',
                              '{}_hip_link_{}']:
                        link_name = s.format(prefix, suffix)
                        link_a = builder.index_from_link_[
                            '{}_leg_origin'.format(prefix)]
                        link_b = builder.index_from_link_[
                            link_name]
                        pb.setCollisionFilterPair(
                            phonebot_id, phonebot_id, link_a, link_b, 1,
                            physicsClientId=sim_id)

        # Optionally create a constraint to fix the robot mid-air.
        if self.settings.debug_midair:
            cid = pb.createConstraint(
                phonebot_id, -1, -1, -1,
                pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.2],
                physicsClientId=sim_id)

        # Optionally add debug visualization.
        # NOTE(yycho0108): Only enabled if render is enabled.
        if self.settings.debug and self.settings.render:
            # Frame axes
            axis_scale = self.settings.debug_axis_scale
            debug_draw_frame_axes(sim_id, phonebot_id, axis_scale)

            # Robot name text
            # pb.addUserDebugText('phonebot', [0, 0, 0.2], [
            # 0, 0, 0], parentObjectUniqueId=phonebot_id,
            # parentLinkIndex=builder.index_from_link_['body'],
            # physicsClientId=sim_id)

            # Inertia
            if self.settings.debug_inertia:
                for i in builder.link_from_index_.keys():
                    debug_draw_inertia_box(phonebot_id, i, [1, 0, 0])

        # Configure whether to show the default pybullet GUI.
        if self.settings.render:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_GUI,
                self.settings.show_pybullet_gui,
                physicsClientId=sim_id)

        # Optionally record video.
        if self.settings.record_file:
            pb.startStateLogging(
                pb.STATE_LOGGING_VIDEO_MP4,
                self.settings.record_file,
                physicsClientId=sim_id)

    def seed(self, seed=None):
        # TODO(ycho): self.task_.seed(seed) or self.task_.set_rng(self.np_random)
        # Perhaps it would be better to prefer the latter option? to avoid
        # parallel `np_random` RNG instances within the same env.
        self.np_random, seed = seeding.np_random(seed)
        self.task_.set_rng(self.np_random)
        return [seed]

    def reset(self) -> np.ndarray:
        """Reset joint values to approximately nominal stance.

        Returns:
            np.ndarray: Sensor sense vector concatenated with the target vector
        """
        # FIXME(yycho0108): store nominal_joint_angles in `PhonebotSettings`.
        config = self.config
        builder = self.builder
        phonebot_id = self.phonebot_id
        sim_id = self.sim_id

        # Reset time / step counter.
        self.time_index = 0
        self.time = 0.0
        self.last_action_time_ = 0
        self.last_action_ = np.zeros(8)

        # Set timestep.
        if self.settings.random_timestep:
            self.timestep = self.np_random.uniform(
                self.settings.timestep_range[0],
                self.settings.timestep_range[1])
            # print('timestep = {}'.format(self.timestep))
        else:
            self.timestep = self.settings.timestep

        pb.setTimeStep(self.timestep, physicsClientId=sim_id)

        # Reset joint states.
        for prefix in config.order:
            for suffix in 'ab':
                index = builder.index_from_joint_[
                    '{}_hip_joint_{}'.format(prefix, suffix)]
                pb.resetJointState(
                    phonebot_id,
                    index,
                    config.nominal_hip_angle,
                    physicsClientId=sim_id)
                index = builder.index_from_joint_[
                    '{}_knee_joint_{}'.format(prefix, suffix)]
                pb.resetJointState(
                    phonebot_id,
                    index,
                    config.nominal_knee_angle,
                    physicsClientId=sim_id)

        # Set camera position and orientation.
        # TODO(yycho0108): Consider exposing these settings.
        if self.settings.render:
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=-0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0.0],
                physicsClientId=sim_id)

        # Apply default reset to base position.
        # NOTE(ycho): Base lin/ang vels are also set to zero.
        pb.resetBasePositionAndOrientation(
            self.phonebot_id, [0, 0, 0], [0, 0, 0, 1], physicsClientId=sim_id)

        # Reset pid controllers as well>
        self.pid_.reset()

        # Reset sensors.
        self._sensor.reset(sim_id, phonebot_id)

        # NOTE(ycho): Task will be allowed to reset the current state as well.
        # Therefore, explicitly return the result from self.sense() after
        # task.reset().
        self.task_.reset(sim_id, phonebot_id)

        # If intersecting with the plane, lift base up to avoid collision.
        # NOTE(ycho): Disabled for now to prevent breaking the pre-trained
        # agent.
        if True:
            robot_aabb = debug_get_full_aabb(sim_id, phonebot_id)
            if robot_aabb[0, 2] < 0:
                #print('robot_aabb[0,2] = {}'.format(robot_aabb[0, 2]))
                #logger.warn('Rectifying collision')
                pos, rot = pb.getBasePositionAndOrientation(
                    phonebot_id, physicsClientId=sim_id)
                pos = np.asarray(pos)
                # NOTE(ycho): assumes plane z == 0.
                # equally valid would be (plane_aabb[1,2] - robot_aabb[0,2])
                pos[2] -= 1e-4 + robot_aabb[0, 2]
                pb.resetBasePositionAndOrientation(
                    phonebot_id, pos, rot, physicsClientId=sim_id)

        # NOTE(ycho): Mostly intended to disable the default pybullet motor,
        # but if `joint_friction_force` is nonzero, it would add damping to the
        # motors.
        if True:
            joint_friction_force = self.settings.joint_friction_force
            all_joint_indices = [builder.index_from_joint_[joint]
                                 for joint in self.config.joint_names]
            pb.setJointMotorControlArray(
                phonebot_id,
                all_joint_indices,
                pb.VELOCITY_CONTROL,
                targetVelocities=[0 for _ in all_joint_indices],
                forces=[joint_friction_force for _ in all_joint_indices],
                physicsClientId=sim_id)

        # FIXME(ycho): Specific hack for ReachPositionTask debug visualization.
        # TODO(ycho): Migrate this code block inside ReachPositionTask.reset().
        if isinstance(
                self.task_, ReachPositionTask) and self.settings.debug and self.settings.render:
            if not self.task_debug_ids_:
                self.task_debug_ids_ = [None for _ in range(3)]
            # FIXME(yycho0108): Temporary visualization
            I = np.eye(3)
            o = [self.task_.target_x, self.task_.target_y, 0.0]
            q = pb.getQuaternionFromEuler([0, 0, self.task_.target_h])
            for i, ax in enumerate(I):
                d = np.float32(pb.rotateVector(
                    q, 4 * ax * self.settings.debug_axis_scale))

                item_id = self.task_debug_ids_[
                    i] if self.task_debug_ids_[i] else -1
                self.task_debug_ids_[i] = pb.addUserDebugLine(
                    o, o + d, ax, parentObjectUniqueId=-1, parentLinkIndex=-1,
                    physicsClientId=sim_id, replaceItemUniqueId=item_id)

                # self.task_text_id_ = pb.addUserDebugText('task-point', [0, 0, 0.2], [0, 0, 0],
                #                                     parentObjectUniqueId=-1,
                #                                     parentLinkIndex=-1)
        if self.settings.debug_servo:
            self.skpi = pb.addUserDebugParameter(
                'kp', 0.0, 1.0, self.settings.servo_kp, physicsClientId=sim_id)
            self.skii = pb.addUserDebugParameter(
                'ki', 0.0, 10.0, self.settings.servo_ki,
                physicsClientId=sim_id)
            self.skdi = pb.addUserDebugParameter(
                'kd', -0.5, 0.5, 10.0 * self.settings.servo_kd,
                physicsClientId=sim_id)

        return self.sense()

    def render(self, mode='human'):
        # logger.warn(
        #    'render() ignored; render mode set in PybulletSimulatorSettings.render'
        #    'cannot be changed during runtime.')
        return

    def sense(self) -> np.ndarray:
        """Return the sensor readings and the task target

        Returns:
            np.ndarray: A vector of the sensor reading concatenated with the
                task target.
        """
        return np.r_[self._sensor.sense(), self.task_.target]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the simulation and take the given action

        Args:
            action (np.ndarray): An array of Phonebot actions to take

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A Tuple comprising:
                (np.ndarray): The current state vector
                (float): The reward for the current state
                (bool): True if Done, False otherwise
                (Dict[str, Any]): Extra info dictionary.
        """

        # Bring relevant params into current scope.
        config = self.config
        builder = self.builder
        phonebot_id = self.phonebot_id
        sim_id = self.sim_id

        if self.settings.debug_servo:
            kp = pb.readUserDebugParameter(
                self.skpi, physicsClientId=self.sim_id)
            ki = pb.readUserDebugParameter(
                self.skii, physicsClientId=self.sim_id)
            kd = (1.0 / 10.0) * pb.readUserDebugParameter(
                self.skdi, physicsClientId=self.sim_id)
            # FIXME(ycho): max_i set to something that feels reasonable ...
            # hardcoded.
            max_i = (4 * self.servo_torque / ki if ki > 0 else 0)
            self.pid_.set_gains(kp, ki, kd)
            self.pid_.set_max_i(max_i)
            self.pid_.reset()

        # FIXME(ycho): Temporary hack for placing 0 @ nominal stance position.
        if self.settings.normalize_action_space:
            # map (-1,1) -> (-np.pi/2,np.pi/2)
            action = action * np.pi / 2
        if self.settings.zero_at_nominal:
            action = np.asarray(action)
            action += config.nominal_hip_angle

        # Get current position.
        # TODO(ycho): Consider caching previous position instead.
        state0 = self.sense()

        joint_indices = [builder.index_from_joint_[joint]
                         for joint in self.config.active_joint_names]

        if not self.settings.use_torque_control:
            # NOTE(ycho):
            # POSITION_CONTROL is implemented via constraints.
            # This one has the advantage of simplicity,
            # but correctness is a bit of a question mark ... and
            # much more iterations are required for convergence.
            servo_force = self.servo_torque
            pb.setJointMotorControlArray(
                phonebot_id,
                joint_indices,
                pb.POSITION_CONTROL,
                targetPositions=action,
                forces=[servo_force for _ in joint_indices],
                physicsClientId=sim_id
            )
        else:
            # NOTE(ycho):
            # TORQUE_CONTROL is implemented via external forces.
            # This one has a disadvantage that it relies on our own
            # (potentially unreliable) PID controller,
            # but has the advantage that much fewer iterations are required.

            # NOTE(ycho): Model servo duty cycle ??
            if False:
                if self.time >= self.last_action_time_ + 0.020:
                    self.last_action_time_ = self.time
                    self.last_action_ = action
                action = self.last_action_

            joint_states = pb.getJointStates(
                phonebot_id,
                joint_indices,
                physicsClientId=sim_id,
            )
            joint_positions = np.array(
                [joint[0] for joint in joint_states]
            )
            position_error = anorm(action - joint_positions)
            dt = self.timestep
            torques = self.pid_(position_error, dt)

            # clip torques...
            if self.settings.clip_torque:
                # max_torque = 0.235
                # max_speed = 10.47 rad/s
                # max_torque@speed = max_torque - (max_torque/max_speed) *
                # speed
                joint_velocities = np.array(
                    [joint[1] for joint in joint_states]
                )
                torque_from_speed = self.servo_torque / (10.47)
                max_torque = self.servo_torque - \
                    torque_from_speed * np.abs(joint_velocities)
                max_torque = np.clip(max_torque, 0.0, self.servo_torque)
                old_torques = torques
                torques = np.clip(torques, -max_torque, max_torque)
                # print('jv {}'.format(joint_velocities))
                # print('ot {}'.format(old_torques))
                # print('nt {}'.format(torques))

            pb.setJointMotorControlArray(
                bodyUniqueId=phonebot_id,
                jointIndices=joint_indices,
                controlMode=pb.TORQUE_CONTROL,
                forces=torques,
                physicsClientId=sim_id
            )

        pb_info = pb.stepSimulation(physicsClientId=sim_id)

        if self.settings.debug_solver:
            # TODO(ycho): Prefer print -> logging
            print(pb_info)

        if self.settings.debug_contact:
            # self-collision
            # contacts = pb.getContactPoints(
            #    phonebot_id, phonebot_id, physicsClientId=sim_id)
            # for contact in contacts:
            #    la, lb = builder.link_from_index_[
            #        contact[3]], builder.link_from_index_[contact[4]]
            #    print('contact={},{}'.format(la, lb))
            contacts = pb.getContactPoints(
                self.plane_id, phonebot_id, physicsClientId=sim_id)
            for c in contacts:
                print('{} {}'.format(self.time_index,
                                     builder.link_from_index_[c[4]]))

        if self.settings.render and self.settings.debug_follow_camera:
            bx, bq = pb.getBasePositionAndOrientation(
                phonebot_id, physicsClientId=sim_id)
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.25,
                cameraYaw=-0,
                cameraPitch=-20,
                cameraTargetPosition=bx,
                physicsClientId=sim_id)

        self.time_index += 1
        self.time += self.timestep

        state1 = self.sense()

        reward = self.task_.compute_reward(state0, state1)

        # Add the energy cost term, similar to minitaur env.
        # TODO(ycho): Remove hardcoded parameters
        if True:
            active_indices = [self._index_from_joint(
                j) for j in self.config.active_joint_names]
            # FIXME(ycho): `jt` is INVALID when use_torque_control==True.
            # Therefore, energy cost will be zero ... fix!!
            _, jv, _, jt = zip(*pb.getJointStates(
                phonebot_id, active_indices, physicsClientId=sim_id))

            if False:
                joint_states = pb.getJointStates(
                    phonebot_id,
                    joint_indices,
                    physicsClientId=sim_id,
                )
                joint_positions = np.array(
                    [joint[0] for joint in joint_states]
                )
                err = anorm(action - joint_positions)

                with open('/tmp/torque_data.txt', 'a') as f:
                    s = np.array2string(np.r_[err, jt],
                                        max_line_width=np.inf,
                                        precision=8,
                                        separator=' ')[1:-1]
                    f.write('{}\n'.format(s))

            max_energy_cost = 8 * 5.0 * 0.65 * self.timestep
            energy_cost = np.abs(np.dot(jv, jt)) * self.timestep
            energy_cost /= max_energy_cost
            # print('energy cost = {}/{}'.format(0.01 * energy_cost, reward))
            reward -= 0.1 * energy_cost

        task_done = self.task_.compute_is_done(state1)

        done = (
            # max time
            (self.time_index >= self.settings.max_num_steps) or
            # reached target position
            task_done
        )

        if self.settings.render and self.settings.debug_show_trajectory:
            x0 = state0[self._sensor.slice_base_position][:3].copy()
            x1 = state1[self._sensor.slice_base_position][:3].copy()

            # Project to ground for a prettier visualization.
            # NOTE(ycho): Workaround until we train a new agent.
            x0[2] = 0  # -0.04
            x1[2] = 0  # -0.04

            pb.addUserDebugLine(x0, x1, [0.0, 0.0, 1.0],
                                lifeTime=1.0,
                                lineWidth=4.0,
                                parentObjectUniqueId=-1,
                                parentLinkIndex=-1,
                                physicsClientId=sim_id)

        info = {}  # TODO(yycho0108): Return something from this.
        return [state1, reward, done, info]

    def close(self):
        # Valid connection id is positive, per pybullet documentation.
        if self.sim_id > 0:
            pb.disconnect(self.sim_id)
        self.sim_id = -1


register(
    id='phonebot-pybullet-v0',
    entry_point='pyphonebot_extra.sim.pybullet.simulator:PybulletPhonebotEnv',
    kwargs={'sim_settings': None, 'phonebot_settings': None}
)

register(
    id='phonebot-pybullet-realtime-v0',
    entry_point='pyphonebot_extra.sim.pybullet.simulator:PybulletPhonebotEnv',
    kwargs={'sim_settings': PybulletSimulatorSettings(
        realtime=True), 'phonebot_settings': None}
)

register(
    id='phonebot-pybullet-headless-v0',
    entry_point='pyphonebot_extra.sim.pybullet.simulator:PybulletPhonebotEnv',
    kwargs={'sim_settings': PybulletSimulatorSettings(
        render=False), 'phonebot_settings': None}
)

# NOTE(ycho): Disabling subprocessing proxy until better package stucture is found
#from pyphonebot_extra.sim.common.subproc import subproc
# @subproc
# class PybulletPhonebotSubprocEnv(PybulletPhonebotEnv):
#    pass
#
# register(
#    id='phonebot-pybullet-headless-subproc-v0',
#    entry_point='pyphonebot_extra.sim.pybullet.simulator:PybulletPhonebotSubprocEnv',
#    kwargs={'sim_settings': PybulletSimulatorSettings(
#        render=False), 'phonebot_settings': None}
# )
