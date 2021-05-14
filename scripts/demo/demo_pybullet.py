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
from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer.viewer_base import HandleHelper
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_handlers import LineStripHandler

from phonebot.sim.common.model import *
from phonebot.sim.common.phonebot_model import PhonebotModel
from phonebot.sim.pybullet.urdf_editor import export_urdf
from phonebot.sim.pybullet.builder import PybulletBuilder
from phonebot.sim.pybullet.simulator import PybulletPhonebotEnv, PybulletSimulatorSettings

from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from phonebot.app.app_utils import update_settings_from_arguments


class AppSettings(Settings):
    """App settings for running pybullet simulator with phonebot."""
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
        viewer = PhonebotViewer()
        handler = HandleHelper(viewer)
        for leg_prefix in config.order:
            viewer.register('{}_trajectory'.format(leg_prefix),
                            LineStripHandler)
            viewer.register('{}_target'.format(leg_prefix),
                            LineStripHandler)

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
            with handler.collect():
                handler.poses(poses=poses)
                handler.edges(poses=poses, edges=edges)

                # Collect current foot positions.
                for leg_prefix in config.order:
                    foot_joint = '{}_foot_a'.format(leg_prefix)
                    foot_positions[leg_prefix].append(
                        graph.get_transform(
                            foot_joint, 'local', stamp).position)

                # Compute target foot positions, on local frame.
                for leg_prefix in config.order:
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
                    handler[tag](pos=traj_points, color=traj_color)

                for leg_prefix in config.order:
                    tag = '{}_target'.format(leg_prefix)
                    traj_points = np.asarray(target_positions[leg_prefix])
                    traj_color = (1.0, 1.0, 0.0, 1.0)
                    handler[tag](pos=traj_points, color=traj_color)
    env.close()


if __name__ == '__main__':
    main()
