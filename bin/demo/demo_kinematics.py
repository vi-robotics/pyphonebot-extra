#!/usr/bin/env python3

import numpy as np
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.core.common.queue_listener import QueueListener
from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import solve_inverse_kinematics, solve_knee_angle, get_graph_geometries
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.transform import Position
from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer.proxy_command import ProxyCommand
from phonebot.vis.viewer.proxy_commands import AddPointsCommand

class PickableViewer(PhonebotViewer):
    def __init__(self, *args, **kwargs):
        self.select_index = None
        self.mx = 0
        self.my = 0
        self.pxs = 1.0
        super().__init__(*args, **kwargs)

    def on_mouse(self, event, event_type):
        if event_type == 'press':
            self.select_index = None
            if event.button() not in [QtCore.Qt.RightButton]:
                return

            # Project all items to screen space.
            pos3d = self.items_['foot_ctrl'].pos
            P = pg.transformToArray(self.widget_.projectionMatrix() * self.widget_.viewMatrix())
            p = (pos3d @ P[:3,:3].T + P[:3,3:].T)
            p = (p[:, :2] / p[:, 2:])
            p = (p + 1.0) / 2.0
            p[:,1] = 1.0 - p[:,1]
            px, py = self.widget_.width() * p[:,0],  self.widget_.height() * p[:,1], 

            # Project cursor position.
            pos = self.widget_.mapFromGlobal(event.globalPos())
            mx, my = pos.x(), pos.y()
            self.mx = mx
            self.my = my

            delta = np.linalg.norm(np.stack([px, py], axis=-1) - (mx,my), axis=-1)
            sel = np.argmin(delta)
            print(delta[sel])
            if delta[sel] < 5.0:
                world_from_camera = pg.transformToArray(self.widget_.viewMatrix())
                R = world_from_camera[:3,:3].T
                xvec = R.dot([-1,0,0])
                yvec = R.dot([0,-1,0])
                # zvec = R.dot([0,0,1])

                self.select_index = sel
                px = self.widget_.pixelSize(self.items_['foot_ctrl'].pos[self.select_index])
                self.xvec = xvec * px
                self.yvec = yvec * px
                self.pold = self.items_['foot_ctrl'].pos[self.select_index].copy()
        elif event_type == 'move':
            if self.select_index is None:
                return

            pos = self.widget_.mapFromGlobal(event.globalPos())
            mx, my = pos.x(), pos.y()

            dx = (-(mx - self.mx) * self.xvec)
            dy = ((my - self.my) * self.yvec)
            pold = self.pold
            pnew = self.pold + dx + dy
            if not self.event_queue_.full():
                self.event_queue_.put_nowait([self.select_index, pold, pnew])
        elif event_type == 'release':
            self.select_index = None

        return

        # print(pos3d)
        # img = self.widget_.readQImage()
        img = self.widget_.renderToArray(
                (self.widget_.width(), self.widget_.height())).swapaxes(0,1)
        pos = self.widget_.mapFromGlobal(event.globalPos())
        x, y = (pos.x(), pos.y())
        h, w = img.shape[:2]

        arr  = img

        # event.accept()
        if not self.event_queue_.full():
            self.event_queue_.put_nowait(arr)


def update_angles(graph: PhonebotGraph, hip_angle_a: float, hip_angle_b: float, stamp: float, config: PhonebotSettings):
    # Initialize angles to 0.
    for leg_prefix in config.order:
        leg_origin = '{}_leg_origin'.format(leg_prefix)
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
        print('knee : {}'.format(knee_angle_a))

        # Set knee
        graph.get_edge(foot_a, knee_joint_a).update(
            stamp, knee_angle_a)
        graph.get_edge(foot_b, knee_joint_b).update(
            stamp, knee_angle_b)


def main():
    global hip_angle_a
    global hip_angle_b
    config = PhonebotSettings()
    config.queue_size = 1
    graph = PhonebotGraph(config)
    data_queue, event_queue, command_queue = PickableViewer.create()

    command_queue.put(AddPointsCommand(name='foot_ctrl'))

    def on_event(event):
        global hip_angle_a
        global hip_angle_b
        try:
            idx, pold, pnew = event
            stamp = time.time()
            pnew[1] = pold[1]
            leg_from_local = graph.get_transform('local', '{}_leg_origin'.format(config.order[idx]), stamp)
            pnew_leg = leg_from_local * Position(pnew)
            ja, jb = solve_inverse_kinematics(
                    graph, stamp, config.order[idx],
                    pnew_leg, config)
            hip_angle_a = ja
            hip_angle_b = jb
            print(ja, jb)
            # update_angles(graph, ja, jb, stamp, config)
        except Exception as e:
            print(e)

        #if isinstance(event, np.ndarray):
        #    cv2.imshow('win', event)
        #    cv2.waitKey(0)
        #print('!!!!!!!!!!!!!!!!!!!', type(event))
        return

    mouse_listener = QueueListener(event_queue, on_event)
    mouse_listener.start()

    # Arbitrary stamp.
    stamp = time.time()
    hip_angle_a = config.nominal_hip_angle
    hip_angle_b = config.nominal_hip_angle
    update_angles(graph, hip_angle_a, hip_angle_b, stamp, config)

    colors = {
        'FL': (0, 0, 1, 1),
        'FR': (0, 1, 0, 1),
        'HL': (1, 0, 0, 1),
        'HR': (1, 0, 1, 1)
        }

    # Sweep angles for both joints, run ik and visualize results.
    # for hip_angle_a in np.linspace(0.0, 2*np.pi, 20):
    #    for hip_angle_b in np.linspace(0.0, 2*np.pi, 20):

    while True:
        stamp = time.time()
        update_angles(graph, hip_angle_a, hip_angle_b, stamp, config)

        # Send data to asynchronous viewer.
        poses, edges = get_graph_geometries(graph, stamp, tol=np.inf)
        foot_pos = np.float32([poses['{}_foot_a'.format(
            prefix)].position for prefix in config.order])
        foot_col = np.float32([colors[prefix] for prefix in config.order])
        if not data_queue.full():
            data_queue.put_nowait(
                {
                    'poses': dict(poses=poses),
                    'edges': dict(poses=poses, edges=edges),
                    'foot_ctrl': dict(pos=foot_pos, color=foot_col)
                })


if __name__ == '__main__':
    main()
