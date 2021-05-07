#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import time
import numpy as np
from collections import deque, defaultdict

from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.math.transform import Rotation, Position, Transform
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.settings import Settings
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import get_graph_geometries, solve_knee_angle, solve_inverse_kinematics
from phonebot.core.kinematics.workspace import get_workspace
from phonebot.core.controls.controllers.base_rotation_controller import BaseRotationController
from phonebot.core.controls.agents.trajectory_agent import TrajectoryAgentGraph
from phonebot.core.common.logger import set_log_level, get_default_logger

from phonebot.vis.viewer import PhonebotViewer
from phonebot.vis.viewer.proxy_commands import AddLineStripCommand

from phonebot.app.app_utils import update_settings_from_arguments


class AppSettings(Settings):
    """
    App settings for Phonebot walking trajectory demonstration.
    """
    robot: PhonebotSettings
    acceleration: float
    log_level: str

    def __init__(self, **kwargs):
        self.robot = PhonebotSettings()
        self.acceleration = 1.0
        self.log_level = 'WARN'
        super().__init__(**kwargs)


def main():
    # Setup settings.
    settings = AppSettings()
    update_settings_from_arguments(settings)
    set_log_level(settings.log_level)
    acceleration = settings.acceleration
    config = settings.robot

    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PhonebotViewer.create()

    for leg_prefix in config.order:
        command_queue.put(AddLineStripCommand(
            name='{}_target'.format(leg_prefix)))

    # Initialize with current time.
    stamp = time.time() * acceleration
    # graph.get_edge('body', 'local').update(0.0,
    #    Transform.from_position(Position([0, 0, 0.1])))

    # Initialize angles to nominal stance.
    for leg_prefix in config.order:
        leg_origin = '{}_leg_origin'.format(leg_prefix)
        hip_joint_a = '{}_hip_joint_a'.format(leg_prefix)
        hip_joint_b = '{}_hip_joint_b'.format(leg_prefix)
        knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
        knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)
        foot_a = '{}_foot_a'.format(leg_prefix)
        foot_b = '{}_foot_b'.format(leg_prefix)

        graph.get_edge(knee_joint_a, hip_joint_a).update(
            stamp, config.nominal_hip_angle)
        graph.get_edge(foot_a, knee_joint_a).update(
            stamp, config.nominal_knee_angle)
        graph.get_edge(knee_joint_b, hip_joint_b).update(
            stamp, config.nominal_hip_angle)
        graph.get_edge(foot_b, knee_joint_b).update(
            stamp, config.nominal_knee_angle)

    agent = TrajectoryAgentGraph(graph, 2.0, config)

    # Obtain joint edges in order.
    joint_edges = []
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            knee = '{}_knee_joint_{}'.format(leg_prefix, leg_suffix)
            hip = '{}_hip_joint_{}'.format(leg_prefix, leg_suffix)
            joint_edge = graph.get_edge(knee, hip)
            joint_edges.append(joint_edge)

    workspace = get_workspace()

    lines = []
    old_foot_positions = []
    target_positions = defaultdict(lambda: deque(maxlen=128))
    state = None
    for _ in range(10000):
        stamp = time.time() * acceleration
        commands = agent(state, stamp)
        foot_positions = []

        # Acutation. Currently, the exact joint value is written.
        # ( "Perfect" joint )
        if True:
            for joint_edge, joint_command in zip(joint_edges, commands):
                joint_edge.update(stamp, anorm(
                    alerp(joint_edge.angle, joint_command, 1.0)))
                # joint_edge.update(stamp, joint_command)
                # joint_edge.update(stamp, joint_edge.angle + 0.01)

            # Update passive joints accordingly here as well.
            for leg_prefix in config.order:
                knee_angle_a, knee_angle_b = solve_knee_angle(
                    graph, leg_prefix, stamp, config=config)

                foot_a = '{}_foot_a'.format(leg_prefix)
                foot_b = '{}_foot_b'.format(leg_prefix)
                knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
                knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)

                knee_edge_a = graph.get_edge(foot_a, knee_joint_a)
                knee_edge_b = graph.get_edge(foot_b, knee_joint_b)

                knee_edge_a.update(stamp, alerp(
                    knee_edge_a.angle, knee_angle_a, 0.5))
                knee_edge_b.update(stamp, alerp(
                    knee_edge_b.angle, knee_angle_b, 0.5))

        # Query endpoint positions.
        for leg_prefix in config.order:
            foot_joint = '{}_foot_a'.format(leg_prefix)
            foot_positions.append(graph.get_transform(
                foot_joint, 'body', stamp).position)

        for leg_prefix in config.order:
            # Compute target foot position on local frame.
            target_foot_leg_origin = agent.trajectories_[
                leg_prefix].evaluate(stamp)
            target_foot_local = graph.get_transform(
                '{}_leg_origin'.format(leg_prefix),
                'local', stamp) * target_foot_leg_origin

            # Append to trajectories.
            target_positions[leg_prefix].append(target_foot_local)

        # NOTE(yycho0108): this is a hack.
        if len(old_foot_positions) > 0:
            for prv, nxt in zip(old_foot_positions, foot_positions):
                lines.append([prv, nxt])
        else:
            for cur in foot_positions:
                lines.append([cur, cur])

        # NOTE(yycho0108): this is a hack.
        # adding workspace visualization.
        ws_lines = []
        for leg_prefix in config.order:
            body_from_leg = graph.get_transform(
                '{}_leg_origin'.format(leg_prefix), 'body', stamp)
            init = False
            for bound_point in workspace:
                p = body_from_leg * bound_point
                if not init:
                    init = True
                    ws_lines.append([p, p])
                else:
                    ws_lines.append([ws_lines[-1][-1], p])

        old_foot_positions = foot_positions

        # Send data to asynchronous viewer.
        poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
        if not data_queue.full():
            trajectory_lines = lines[-100:]
            trajectory_colors = np.broadcast_to(
                [0, 1, 1], np.shape(trajectory_lines)).reshape(-1, 3)
            workspace_lines = ws_lines
            workspace_colors = np.broadcast_to(
                [1, 1, 0], np.shape(workspace_lines)).reshape(-1, 3)
            extra_lines = np.concatenate(
                [trajectory_lines, workspace_lines], axis=0)
            extra_colors = np.concatenate(
                [trajectory_colors, workspace_colors], axis=0)

            visdata = {
                'poses': dict(
                    poses=poses), 'edges': dict(
                    poses=poses, edges=edges), 'line': dict(
                    pos=extra_lines, color=extra_colors)}
            for leg_prefix in config.order:
                tag = '{}_target'.format(leg_prefix)
                traj_points = np.asarray(target_positions[leg_prefix])
                traj_color = (1.0, 0.0, 1.0, 1.0)
                visdata[tag] = dict(pos=traj_points, color=traj_color)

            data_queue.put_nowait(visdata)


if __name__ == '__main__':
    main()
