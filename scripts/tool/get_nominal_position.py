#!/usr/bin/env python3

import numpy as np
import time

from phonebot.core.common.queue_listener import QueueListener
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics, solve_knee_angle, get_graph_geometries
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.kinematics.workspace import get_workspace, max_rect
from phonebot.core.common.math.transform import Position


def update_angles(graph: PhonebotGraph, hip_angle_a: float,
                  hip_angle_b: float, stamp: float, config: PhonebotSettings):
    # Initialize angles to 0.
    for leg_prefix in config.order:
        leg_origin = '{}_leg_origin'.format(leg_prefix)
        hip_joint_a = '{}_hip_joint_a'.format(leg_prefix)
        hip_joint_b = '{}_hip_joint_b'.format(leg_prefix)
        knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
        knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)
        foot_a = '{}_foot_a'.format(leg_prefix)
        foot_b = '{}_foot_b'.format(leg_prefix)

        # Set Hip angle
        graph.get_edge(knee_joint_a, hip_joint_a).update(
            stamp, hip_angle_a)
        graph.get_edge(knee_joint_b, hip_joint_b).update(
            stamp, hip_angle_b)

        knee_angle_a, knee_angle_b = solve_knee_angle(
            graph, leg_prefix, stamp, config)
        print('knee : {}'.format(knee_angle_a))

        # Set knee angle
        graph.get_edge(foot_a, knee_joint_a).update(
            stamp, knee_angle_a)
        graph.get_edge(foot_b, knee_joint_b).update(
            stamp, knee_angle_b)


def main():
    config = PhonebotSettings()
    config.queue_size = 1
    graph = PhonebotGraph(config)
    # update_angles(graph, 0.0, 0.0, 0.0, config)
    body_from_leg = graph.get_transform(
        'FL_leg_origin', 'body', 0.0, tol=np.inf)

    workspace = get_workspace(0.0, config, return_poly=True)
    ws_rect = max_rect(workspace, 4096)
    (x0, y0), (x1, y1) = ws_rect
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    foot = body_from_leg * Position([cx, cy, 0.0])
    hip_a, hip_b = solve_inverse_kinematics(graph, 0.0, 'FL', foot, config)
    graph.get_edge('FL_knee_joint_a', 'FL_hip_joint_a').update(0.0, hip_a)
    graph.get_edge('FL_knee_joint_b', 'FL_hip_joint_b').update(0.0, hip_b)
    knee_a, knee_b = solve_knee_angle(graph, 'FL', 0.0, config, [cx, cy])
    print('hip : {} {} mean={}'.format(hip_a, hip_b, np.mean([hip_a, hip_b])))
    print('knee : {} {} mean={}'.format(
        knee_a, knee_b, np.mean([knee_a, knee_b])))


if __name__ == '__main__':
    main()
