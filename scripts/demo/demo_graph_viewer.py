#!/usr/bin/env python3

import time
import numpy as np

from phonebot.core.common.math.utils import anorm
from phonebot.core.common.config import PhonebotSettings
from phonebot.vis.viewer import PhonebotViewer
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import solve_knee_angle, solve_inverse_kinematics, get_graph_geometries


def main():
    config = PhonebotSettings()
    # Set queue size to 1 since we're not changing time.
    # (This will get rid of buffering artifacts)
    config.queue_size = 1
    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PhonebotViewer.create()

    # Arbitrary stamp.
    stamp = time.time()

    # Initialize angles to 0.
    for leg_prefix in config.order:
        leg_origin = '{}_leg_origin'.format(leg_prefix)
        hip_joint_a = '{}_hip_joint_a'.format(leg_prefix)
        hip_joint_b = '{}_hip_joint_b'.format(leg_prefix)
        knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
        knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)
        foot_a = '{}_foot_a'.format(leg_prefix)
        foot_b = '{}_foot_b'.format(leg_prefix)

        graph.get_edge(knee_joint_a, hip_joint_a).update(
            stamp, 0.0)
        graph.get_edge(foot_a, knee_joint_a).update(
            stamp, 0.0)
        graph.get_edge(knee_joint_b, hip_joint_b).update(
            stamp, 0.0)
        graph.get_edge(foot_b, knee_joint_b).update(
            stamp, 0.0)

    # Sweep angles for both joints, run ik and visualize results.
    for hip_angle_a in np.linspace(0.0, 2*np.pi, 20):
        for hip_angle_b in np.linspace(0.0, 2*np.pi, 20):

            for leg_prefix in config.order:
                leg_origin = '{}_leg_origin'.format(leg_prefix)
                hip_joint_a = '{}_hip_joint_a'.format(leg_prefix)
                hip_joint_b = '{}_hip_joint_b'.format(leg_prefix)
                knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
                knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)
                foot_a = '{}_foot_a'.format(leg_prefix)
                foot_b = '{}_foot_b'.format(leg_prefix)

                graph.get_edge(knee_joint_a, hip_joint_a).update(
                    stamp, hip_angle_a)
                graph.get_edge(knee_joint_b, hip_joint_b).update(
                    stamp, hip_angle_b)
                knee_angle_a, knee_angle_b = solve_knee_angle(
                    graph, leg_prefix, stamp, config=config)

                stamp = time.time()
                graph.get_edge(knee_joint_a, hip_joint_a).update(
                    stamp, hip_angle_a)
                graph.get_edge(knee_joint_b, hip_joint_b).update(
                    stamp, hip_angle_b)
                graph.get_edge(foot_a, knee_joint_a).update(
                    stamp, knee_angle_a)
                graph.get_edge(foot_b, knee_joint_b).update(
                    stamp, knee_angle_b)

                pos_a = graph.get_transform(
                    foot_a, 'body', stamp).position
                pos_b = graph.get_transform(
                    foot_b, 'body', stamp).position
                print('foot_positions : {} == {}'.format(pos_a, pos_b))
                ik_solution = solve_inverse_kinematics(
                    graph, stamp, leg_prefix, pos_a, config=config)
                print('angles : {} == {}'.format(
                    anorm([hip_angle_a, hip_angle_b]), ik_solution))

                # Send data to asynchronous viewer.
                poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
                if not data_queue.full():
                    data_queue.put_nowait(
                        {'poses': dict(poses=poses), 'edges': dict(poses=poses, edges=edges)})


if __name__ == '__main__':
    main()
