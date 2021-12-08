#!/usr/bin/env python3
"""
Inverse kinematics demo, for moving phonebot legs to a designated position
as specified by the user (through mouse `pick` event).
All standard controls for GLViewWidget applies, plus RMB for picking.
"""

import numpy as np
import time
import logging

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.core.common.queue_listener import QueueListener
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics, solve_knee_angle, get_graph_geometries
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.transform import Position
from phonebot.vis.viewer._pyqtgraph import PyqtViewer3D, PointsHandler
from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer.viewer_base import HandleHelper



class PickHandler:
    def __init__(self, viewer: PyqtViewer3D, pick_radius: float = 5.0):
        self.viewer = viewer
        self.pick_radius = pick_radius
        self.vertices = None
        self.select_index = None
        # Register mouse event callback.
        self.viewer.on_mouse(self.on_mouse)

    @property
    def widget(self):
        return self.viewer.widget_

    def _on_press(self, data):
        _, button, x, y = data
        # FIXME(ycho): should probably be replaced with MouseEvent(...)
        if button != 2:
            return
        screen_xy = (x, y)
        self.mx = x
        self.my = y

        # Project all items to screen space.
        P = pg.transformToArray(
            self.widget.projectionMatrix() *
            self.widget.viewMatrix())
        p = (self.vertices @ P[:3, :3].T + P[:3, 3:].T)
        p = (p[:, :2] / p[:, 2:])
        p = (p + 1.0) / 2.0
        # FIXME(ycho): Flip Y axis for some reason?
        p[:, 1] = 1.0 - p[:, 1]
        px, py = (self.widget.width() * p[:, 0],
                  self.widget.height() * p[:, 1])

        # Select nearest pixel-space neighbor.
        delta = np.linalg.norm(np.stack([px, py], axis=-1) - screen_xy,
                               axis=-1)
        sel = np.argmin(delta)

        # Apply threshold, etc etc
        if delta[sel] < self.pick_radius:
            world_from_camera = pg.transformToArray(
                self.widget.viewMatrix())
            R = world_from_camera[:3, :3].T
            xvec = R.dot([-1, 0, 0])
            yvec = R.dot([0, -1, 0])

            self.select_index = sel
            pix_size = self.widget.pixelSize(self.vertices[self.select_index])
            self.xvec = xvec * pix_size
            self.yvec = yvec * pix_size
            self.pold = self.vertices[self.select_index].copy()

            self.viewer.trigger('pick', (sel, self.pold, self.pold))

    def _on_move(self, data):
        if self.select_index is None:
            return

        _, _, x, y = data
        mx, my = (x, y)  # maybe ?

        dx = (-(mx - self.mx) * self.xvec)
        dy = ((my - self.my) * self.yvec)
        pold = self.pold
        pnew = self.pold + dx + dy
        # TODO(ycho): figure out why `pold`, etc, should not be updated
        # self.pold = pnew
        self.viewer.trigger('pick', (self.select_index, pold, pnew))

    def _on_release(self, data):
        self.select_index = None


    def on_mouse(self, data):
        # Only process pick-related events if vertices are registered.
        if self.vertices is None:
            return

        # Unpack data.
        event_type, button, x, y = data

        if event_type == 'press':
            return self._on_press(data)
        elif event_type == 'move':
            return self._on_move(data)
        elif event_type == 'release':
            return self._on_release(data)
        else:
            raise KeyError(F'event_type == {event_type} not found')

    def __call__(self, vertices: np.ndarray = None):
        # Update vertices.
        self.vertices = vertices


def update_angles(graph: PhonebotGraph, hip_angle_a: float,
                  hip_angle_b: float, stamp: float, config: PhonebotSettings):
    # Initialize angles to 0.
    for leg_prefix in config.order:
        hip_joint_a = '{}_hip_joint_a'.format(leg_prefix)
        hip_joint_b = '{}_hip_joint_b'.format(leg_prefix)
        knee_joint_a = '{}_knee_joint_a'.format(leg_prefix)
        knee_joint_b = '{}_knee_joint_b'.format(leg_prefix)
        foot_a = '{}_foot_a'.format(leg_prefix)
        foot_b = '{}_foot_b'.format(leg_prefix)

        # Set Hip
        graph.get_edge(knee_joint_a, hip_joint_a).update(
            stamp, hip_angle_a)
        graph.get_edge(knee_joint_b, hip_joint_b).update(
            stamp, hip_angle_b)

        knee_angle_a, knee_angle_b = solve_knee_angle(
            graph, leg_prefix, stamp, config)

        # Set knee
        graph.get_edge(foot_a, knee_joint_a).update(
            stamp, knee_angle_a)
        graph.get_edge(foot_b, knee_joint_b).update(
            stamp, knee_angle_b)


def main():
    logging.basicConfig(level=logging.WARN)
    logging.root.setLevel(level=logging.WARN)

    config = PhonebotSettings()
    config.queue_size = 1
    graph = PhonebotGraph(config)
    viewer = PhonebotViewer()
    viewer.register('foot_ctrl', PointsHandler)
    viewer.register('pick', PickHandler)
    state = {'angles': [0, 0]}

    def on_pick(topic, data):
        try:
            idx, pold, pnew = data
            stamp = time.time()
            pnew[1] = pold[1]
            leg_from_local = graph.get_transform(
                'local', '{}_leg_origin'.format(
                    config.order[idx]), stamp)
            pnew_leg = leg_from_local * Position(pnew)
            ja, jb = solve_inverse_kinematics(
                graph, stamp, config.order[idx],
                pnew_leg, config)
            state['angles'] = [ja, jb]
        except Exception as e:
            logging.error(e)

    viewer.on_event('pick', on_pick)
    handler = HandleHelper(viewer)

    # Initialize from arbitrary stamp.
    stamp = time.time()
    state['angles'] = [config.nominal_hip_angle, config.nominal_hip_angle]
    update_angles(graph, state['angles'][0], state['angles'][1], stamp, config)

    colors = {
        'FL': (0, 0, 1, 1),
        'FR': (0, 1, 0, 1),
        'HL': (1, 0, 0, 1),
        'HR': (1, 0, 1, 1)
    }
    while True:
        stamp = time.time()
        update_angles(graph, state['angles'][0], state['angles'][1],
                      stamp, config)

        # Send data to asynchronous viewer.
        poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
        foot_pos = np.float32([poses['{}_foot_a'.format(
            prefix)].position for prefix in config.order])
        foot_col = np.float32([colors[prefix] for prefix in config.order])

        with handler.collect():
            handler.poses(poses=poses)
            handler.edges(poses=poses, edges=edges)
            handler.foot_ctrl(pos=foot_pos, color=foot_col)
            handler.pick(foot_pos)


if __name__ == '__main__':
    main()
