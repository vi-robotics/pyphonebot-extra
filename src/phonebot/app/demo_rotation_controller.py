#!/usr/bin/env python3

import time
import numpy as np
from collections import defaultdict, deque

import logging
import shapely.ops
from threading import Thread

#from line_profiler import LineProfiler

import pyqtgraph.opengl as gl

from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.math.transform import Rotation, Position, Transform
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.queue_listener import QueueListener
from phonebot.vis.viewer import PhonebotViewer
from phonebot.vis.viewer.proxy_command import ProxyCommand
from phonebot.vis.viewer.proxy_commands import AddLineStripCommand
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import get_graph_geometries, solve_knee_angle, solve_inverse_kinematics
from phonebot.core.kinematics.workspace import get_workspace
from phonebot.core.controls.controllers.base_rotation_controller import BaseRotationController

logger = get_default_logger(logging.WARN)


class KeyControlListener(QueueListener):
    """
    Simple listener remapping keys to actions controlling orientation.
    """

    def __init__(self, queue):
        self.key_map_ = dict(
            w=(0.01, 0.0),
            a=(0.0, -0.01),
            s=(-0.01, 0.0),
            d=(0.0, 0.01),

            q=(0.01, -0.01),
            e=(0.01, 0.01),
            z=(-0.01, -0.01),
            c=(-0.01, 0.01),
            x='quit'
        )
        self.quit_ = False
        self.reset()
        super().__init__(queue, self.on_key)

    def on_key(self, key: str):
        """ Handle keypress input """
        if isinstance(key, int):
            key = chr(key).lower()

        if key not in self.key_map_:
            return
        offsets = self.key_map_[key]
        if offsets == 'quit':
            self.quit_ = True
            self.updated_ = True
            return
        self.offsets_ += offsets
        self.updated_ = True

    def reset(self):
        """ Unset update flag and zero out accumulant."""
        self.updated_ = False
        self.offsets_ = np.zeros(2)

    def has_data(self):
        """ Whether at least one relevant user input was provided """
        return self.updated_

    def get(self):
        """ Retrieve current data and reset state """
        offsets = self.offsets_.copy()
        self.reset()
        return offsets, self.quit_


def main():
    config = PhonebotSettings()
    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PhonebotViewer.create()

    for leg_prefix in config.order:
        command_queue.put(AddLineStripCommand(
            name='{}_workspace'.format(leg_prefix)))
        command_queue.put(AddLineStripCommand(
            name='{}_trajectory'.format(leg_prefix)))

    acceleration = 1.0
    max_iter = np.inf

    # Arbitrary stamp.
    stamp = time.time() * acceleration

    workspace = get_workspace()

    # Initialize to a decent place..
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

    # Obtain joint edges in order.
    joint_edges = []
    for leg_prefix in config.order:
        for leg_suffix in 'ab':
            knee = '{}_knee_joint_{}'.format(leg_prefix, leg_suffix)
            hip = '{}_hip_joint_{}'.format(leg_prefix, leg_suffix)
            joint_edge = graph.get_edge(knee, hip)
            joint_edges.append(joint_edge)

    controller = BaseRotationController(graph)

    plane_visuals = None
    plane_rotation = None

    roll = 0.0
    pitch = 0.0

    while plane_visuals is None:
        # plane_rotation = Rotation.from_euler([0.1,0.1,0])
        plane_rotation = Rotation.from_euler([roll, pitch,
                                              0.0]).to_quaternion()
        plane_normal = plane_rotation.rotate(Position([0, 0, 1]))
        if(plane_normal.dot([0, 0, 1]) < 0):
            continue
        plane_visuals = controller.update(plane_rotation)

    # plane_position = Position(controller.distance_ * controller.normal_)
    plane_position = Position(
        -plane_rotation.inverse().rotate(controller.distance_ * controller.normal_))
    graph.get_edge('body', 'local').update(
        0.0, Transform(plane_position, plane_rotation))

    for edge in graph.edges:
        logger.debug(repr(edge))

    # rotation = Rotation.from_euler([0.2,0.0,0.0])
    # logger.warn(rotation.to_euler())
    # logger.warn(rotation.rotate(Position([0,0,1])))
    # plane_visuals = controller.update(rotation)

    foot_positions = defaultdict(lambda: deque(maxlen=128))
    leg_colors = {k: np.random.uniform(size=3) for k in config.order}

    key_listener = KeyControlListener(event_queue)
    key_listener.start()
    iteration_count = 0
    while True:
        if iteration_count > max_iter:
            break
        stamp = time.time() * acceleration
        logger.info('current stamp : {}'.format(stamp))
        commands = controller.control(stamp)

        # Process key.
        if key_listener.has_data():
            (dp, dr), quit = key_listener.get()
            if quit:
                break
            pitch += dp
            roll += dr

            plane_rotation = Rotation.from_euler([roll, pitch,
                                                  0.0]).to_quaternion()
            plane_visuals = controller.update(plane_rotation)
            plane_position = Position(
                -plane_rotation.inverse().rotate(controller.distance_ * controller.normal_))
            graph.get_edge('body', 'local').update(
                0.0, Transform(plane_position, plane_rotation))

        # Acutation. Currently, the exact joint value is written.
        # ( "Perfect" joint )
        for joint_edge, joint_command in zip(joint_edges, commands):
            joint_edge.update(stamp, anorm(
                alerp(joint_edge.angle, joint_command, 0.5)))

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
            foot_positions[leg_prefix].append(graph.get_transform(
                foot_joint, 'local', stamp).position)

        # TODO: Update local frame view based on foot_positions.?
        graph.get_edge('body', 'local').update(
            0.0, Transform(plane_position, plane_rotation))

        # Compuet workspace visualization data.
        workspace_points = {}
        for leg_prefix in config.order:
            local_from_leg = graph.get_transform(
                '{}_leg_origin'.format(leg_prefix), 'local', stamp)
            workspace_points[leg_prefix] = local_from_leg * Position(workspace)

        plane_lines = []
        plane_colors = []
        for leg_prefix in config.order:
            leg_origin = '{}_leg_origin'.format(leg_prefix)
            foot_joint = '{}_foot_a'.format(leg_prefix)
            body_from_leg = graph.get_transform(leg_origin,
                                                'body', stamp)

            for ps in np.reshape(plane_visuals[leg_prefix], (-1, 2, 3)):
                p_start, p_end = [Position(p) for p in ps]
                plane_lines.append(
                    [body_from_leg*p_start, body_from_leg*p_end])
                plane_colors.extend(
                    [leg_colors[leg_prefix], leg_colors[leg_prefix]])

            ep = graph.get_transform(foot_joint, 'body', stamp).position
            epp = body_from_leg.inverse() * ep

            refs = controller.cache_[leg_prefix]
            cur = shapely.geometry.Point(epp.x, epp.y)

            # test
            source, target = shapely.ops.nearest_points(refs, cur)
            source = Position([source.x, source.y, 0.0])
            target = Position([target.x, target.y, 0.0])
            plane_lines.append([body_from_leg * source, body_from_leg*target])
            plane_colors.extend([[1, 0, 1], [1, 0, 1]])

        # Send data to asynchronous viewer.
        poses, edges = get_graph_geometries(
            graph, stamp, target_frame='local', tol=np.inf)
        if not data_queue.full():
            local_from_body = graph.get_transform('body', 'local', stamp)

            plane_lines = np.float32(plane_lines)
            plane_colors = np.float32(plane_colors)

            extra_lines = np.concatenate(
                [plane_lines], axis=0)
            shape = extra_lines.shape
            extra_lines = local_from_body * \
                Position(extra_lines.reshape(-1, 3))

            extra_lines = np.reshape(extra_lines, shape)
            extra_colors = np.concatenate(
                [plane_colors], axis=0)

            visdata = {'poses': dict(poses=poses), 'edges':
                       dict(poses=poses, edges=edges), 'line':
                       dict(pos=extra_lines,
                            color=extra_colors)}

            # Add workspace visualization.
            for leg_prefix in config.order:
                tag = '{}_workspace'.format(leg_prefix)
                visdata[tag] = dict(
                    pos=workspace_points[leg_prefix],
                    color=(1., 1., 0., 1.))

            # Add endpoint trajectory.
            for leg_prefix in config.order:
                tag = '{}_trajectory'.format(leg_prefix)
                visdata[tag] = dict(
                    pos=np.asarray(foot_positions[leg_prefix]),
                    color=(0., 1., 1., 1.))

            data_queue.put_nowait(visdata)


if __name__ == '__main__':
    main()
    # prof = LineProfiler()
    # prof.add_function(get_graph_geometries)
    # prof.add_function(PhonebotGraph.get_transform)
    # prof(main)()
    # prof.print_stats()
