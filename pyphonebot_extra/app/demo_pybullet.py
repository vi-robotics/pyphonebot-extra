#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from collections import namedtuple, defaultdict, deque
import argparse
import argcomplete
import time
import numpy as np
import gym

from phonebot.core.common.math.transform import Transform, Rotation, Position
from phonebot.core.common.settings import Settings
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph import get_graph_geometries
from phonebot.core.frame_graph.graph_utils import update_passive_joints
from pyphonebot_extra.vis.viewer import PhonebotViewer
from pyphonebot_extra.vis.viewer.proxy_commands import AddLineStripCommand

from pyphonebot_extra.sim.common.model import *
from pyphonebot_extra.sim.common.phonebot_model import PhonebotModel
from pyphonebot_extra.sim.pybullet.urdf_editor import export_urdf
from pyphonebot_extra.sim.pybullet.builder import PybulletBuilder
from pyphonebot_extra.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings

from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from pyphonebot_extra.app.app_utils import update_settings_from_arguments


class AppSettings(Settings):
    """
    App settings for running pybullet simulator with phonebot.
    """
    sim: PybulletSimulatorSettings
    robot: PhonebotSettings
    use_viewer: bool

    def __init__(self, **kwargs):
        self.sim = PybulletSimulatorSettings()
        self.robot = PhonebotSettings()
        self.use_viewer = True
        super().__init__(**kwargs)


def main():
    # Setup settings and arguments.
    settings = AppSettings()
    update_settings_from_arguments(settings)

    # Initialize based on settings / arguments.
    config = settings.robot
    graph = PhonebotGraph(config)

    # Optionally enable inspection through PhonebotViewer.
    if settings.use_viewer:
        data_queue, event_queue, command_queue = PhonebotViewer.create()
        for leg_prefix in config.order:
            command_queue.put(AddLineStripCommand(
                name='{}_trajectory'.format(leg_prefix)))
        for leg_prefix in config.order:
            command_queue.put(AddLineStripCommand(
                name='{}_target'.format(leg_prefix)))
        # For trajectory visualization.
        foot_positions = defaultdict(lambda: deque(maxlen=256))
        target_positions = defaultdict(lambda: deque(maxlen=128))

    # NOTE(yycho0108): Requires PybulletPhonebotEnv import for gym.make to
    # work.
    env = gym.make('phonebot-pybullet-v0', sim_settings=settings.sim,
                   phonebot_settings=settings.robot)
    env.seed(0)
    env.reset()

    agent = TrajectoryAgentGraph(graph, 1.0, config)
    h = 0
    stamp = 0.0

    commands = None
    joint_names = settings.robot.joint_names
    count = -1
    while True:
        count += 1
        # NOTE(yycho0108): Avoid hacky stamp extraction.
        # Possibly make `timestamp` part of the returned state?
        stamp += env.timestep

        # Get current joint states and update the frame graph accordingly.
        # Get environment state and parse the state into a pre-defined order.
        state = env.sense()
        active_joint_states = state[env.sensor.slice_active_joints]
        passive_joint_states = state[env.sensor.slice_passive_joints]
        joint_states = np.zeros(len(config.joint_names), dtype=np.float32)
        joint_states[config.active_joint_indices] = active_joint_states
        joint_states[config.passive_joint_indices] = passive_joint_states
        joint_states_map = {key: value for key, value in zip(
            config.joint_names, joint_states)}
        for prefix in config.order:
            for suffix in 'ab':
                hip_joint = '{}_hip_joint_{}'.format(prefix, suffix)
                knee_joint = '{}_knee_joint_{}'.format(prefix, suffix)
                foot_joint = '{}_foot_{}'.format(prefix, suffix)
                graph.get_edge(knee_joint, hip_joint).update(
                    stamp, joint_states_map[hip_joint])
                graph.get_edge(foot_joint, knee_joint).update(
                    stamp, joint_states_map[knee_joint])

        # Apply control commands from the trajectory controller.
        commands = agent(joint_states, stamp)
        commands = np.array(commands)

        # FIXME(ycho): Hack(s) to normalize action space within (-1,+1).
        if settings.sim.zero_at_nominal:
            commands -= settings.robot.nominal_hip_angle
        if settings.sim.normalize_action_space:
            commands /= np.pi / 2
        out = env.step(commands)

        # [Optional] Also visualize frame graph.
        # TODO(ycho): Refactor out common visualization utilities.
        if settings.use_viewer:
            poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
            visdata = {'poses': dict(poses=poses), 'edges': dict(
                poses=poses, edges=edges)}

            for leg_prefix in config.order:
                foot_joint = '{}_foot_a'.format(leg_prefix)
                foot_positions[leg_prefix].append(
                    graph.get_transform(foot_joint, 'local', stamp).position)

            for leg_prefix in config.order:
                # Compute target foot position on local frame.
                target_foot_leg_origin = agent.trajectories_[
                    leg_prefix].evaluate(stamp)
                target_foot_local = graph.get_transform(
                    '{}_leg_origin'.format(leg_prefix),
                    'local', stamp) * target_foot_leg_origin

                # Append to trajectories.
                target_positions[leg_prefix].append(target_foot_local)

            for leg_prefix in config.order:
                tag = '{}_trajectory'.format(leg_prefix)
                traj_points = np.asarray(foot_positions[leg_prefix])
                traj_color = (0.0, 1.0, 1.0, 1.0)
                visdata[tag] = dict(pos=traj_points, color=traj_color)

            for leg_prefix in config.order:
                tag = '{}_target'.format(leg_prefix)
                traj_points = np.asarray(target_positions[leg_prefix])
                traj_color = (1.0, 1.0, 0.0, 1.0)
                visdata[tag] = dict(pos=traj_points, color=traj_color)

            if not data_queue.full():
                data_queue.put_nowait(visdata)
    env.close()


if __name__ == '__main__':
    main()
