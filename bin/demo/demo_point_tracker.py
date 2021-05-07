#!/usr/bin/env python3

import sys
import zlib
import numpy as np
import time
import logging
import pickle
from typing import List, Dict, Tuple

from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.math.transform import Rotation, Position, Transform
from phonebot.core.common.config import PhonebotSettings, FrameName
from phonebot.core.common.serial import encode, decode
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import (
    get_graph_geometries,
    initialize_graph_nominal,
    initialize_graph_zero,
    update_passive_joints,
    get_joint_edges)
from phonebot.core.controls.agents.point_tracker_agent import PointTrackerAgent

from phonebot.vis.viewer import PhonebotViewer
from phonebot.vis.viewer.proxy_commands import AddPointsCommand, AddLinesCommand
from phonebot.core.common.logger import get_default_logger
logger = get_default_logger(logging.WARN)


def spawn_point() -> Position:
    """ Generate a random point above the phonebot. """
    # NOTE(ycho): The magic numbers below have been tuned
    # to return valid solutions at close proximities.
    # Basically centers the point around the camera FOV.
    return Position(np.random.uniform(
        low=[0.06 + 0.005, 0.02, 0.25],
        high=[0.06 + 0.005, 0.02, 0.25]))


def sense_point(graph: PhonebotGraph, point: Position,
                stamp: float) -> Tuple[float, float]:
    """ Sense the point angular deviation in the camera frame. """
    # Convert to camera frame
    xfm = graph.get_transform(FrameName.LOCAL, FrameName.CAMERA, stamp)
    point = xfm * point
    # Compute angles and return.
    # FIXME(yycho0108): May require negation for frame consistency.
    return (-np.arctan2(point[0], point[2]), -np.arctan2(point[1], point[2]))


def generate_grid():
    """Generate a saved list of commands which correspond to a given
    set of angles.
    """
    # Boilerplate setup ...
    config = PhonebotSettings()
    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PhonebotViewer.create()
    command_queue.put(AddPointsCommand(name='points'))
    command_queue.put(AddLinesCommand(name='lines'))
    stamp = 0.0

    # In the absence of information, initialize to nominal position.
    initialize_graph_nominal(graph, stamp, config)
    agent = PointTrackerAgent(graph, config)

    # Obtain joint edges in order.
    joint_edges = get_joint_edges(graph, config)

    angle_range = .15
    grid_num = 40
    roll_range = np.linspace(-angle_range, angle_range, grid_num)
    pitch_range = np.linspace(-angle_range, angle_range, grid_num)
    rv, pv = np.meshgrid(roll_range, pitch_range, sparse=False, indexing='ij')

    res = []
    state = None
    last_time = time.time()
    for i in range(len(rv)):
        row = []
        for j in range(len(pv)):
            print("Time: {}".format(stamp))
            print("FPS: {}".format(1 / (time.time() - last_time)))
            last_time = time.time()
        stamp += 0.01

        # Update the agent and get commands to control the servos.
        agent.update_target(rv[i, j], pv[i, j], False)
        commands = agent(state, stamp)
        row.append(commands)
        # Simulate interpolated acutation.
        # TODO(yycho0108): How will this be represented in the real hardware?
        # Consider refactoring this part as something like OpenLoopStateUpdater().
        # Alternatively, figure out if these parameters can be solved via some
        # optimization.
        for joint_edge, joint_command in zip(joint_edges, commands):
            joint_edge.update(stamp, anorm(
                alerp(joint_edge.angle, joint_command, 0.5)))
        # Update passive joints based on the above active-joint updates.
        update_passive_joints(graph, stamp, config)

        res.append(row)

    save_data = {
        "commands": res,
        "roll_grid": rv,
        "pitch_grid": pv
    }
    file = open('face_follow_trajectories', 'wb')
    pickle.dump(save_data, file)
    file.close()


def main():
    # Boilerplate setup ...
    config = PhonebotSettings()
    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PhonebotViewer.create()
    command_queue.put(AddPointsCommand(name='points'))
    command_queue.put(AddLinesCommand(name='lines'))
    stamp = 0.0

    # In the absence of information, initialize to nominal position.
    initialize_graph_nominal(graph, stamp, config)
    agent = PointTrackerAgent(graph, config)

    # Obtain joint edges in order.
    joint_edges = get_joint_edges(graph, config)

    # Spawn point to track.
    point_origin = spawn_point()

    state = None
    count = 0
    while True:
        state = encode((stamp, graph.encode()))
        # print(type(state), sys.getsizeof(state), sys.getsizeof(zlib.compress(state)))
        with open('/tmp/state-{:04d}.txt'.format(count), 'wb') as f:
            f.write(zlib.compress(state))
            count += 1

        stamp += 0.01

        # Move the points in a circular arc in a 2s period.
        h = stamp * (2 * np.pi) / 2.0
        s, c = np.sin(h), np.cos(h)
        point = point_origin + 0.025 * Position([s, c, 0])

        # Sense the points angular deviation from the camera and
        # Set the approximate target rotation accordingly.
        # This is kind of like a p controller where you know that
        # The proportional gain coefficient is really close to 1.
        dx, dy = sense_point(graph, point, stamp)

        # Update the agent and get commands to control the servos.
        agent.update_target(dx, dy)
        commands = agent(state, stamp)

        # Simulate interpolated acutation.
        # TODO(yycho0108): How will this be represented in the real hardware?
        # Consider refactoring this part as something like OpenLoopStateUpdater().
        # Alternatively, figure out if these parameters can be solved via some
        # optimization.
        for joint_edge, joint_command in zip(joint_edges, commands):
            joint_edge.update(stamp, anorm(
                alerp(joint_edge.angle, joint_command, 0.5)))
        # Update passive joints based on the above active-joint updates.
        update_passive_joints(graph, stamp, config)

        # Send data to asynchronous viewer.
        # NOTE(yycho0108): All of the following code is concerned with
        # visualization.
        cam_pos = graph.get_transform(
            FrameName.CAMERA, FrameName.LOCAL, stamp).position
        poses, edges = get_graph_geometries(
            graph, stamp, target_frame=FrameName.LOCAL, tol=np.inf)
        if not data_queue.full():
            visdata = {'poses': dict(poses=poses), 'edges':
                       dict(poses=poses, edges=edges),
                       'points': dict(pos=np.float32([point])),
                       'lines': dict(pos=np.float32([[point, cam_pos]]))
                       }
            data_queue.put_nowait(visdata)


if __name__ == '__main__':
    main()
