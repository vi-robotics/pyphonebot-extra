#!/usr/bin/env python3

__all__ = ['PhonebotViewer']

import numpy as np
import pyqtgraph.opengl as gl

from phonebot.core.common.logger import get_default_logger
from phonebot.vis import primitives
from phonebot.vis.viewer import ProxyViewer
from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.core.frame_graph import FrameGraph, StaticFrameEdge, SimpleFrameEdge, RevoluteJointEdge
from phonebot.core.frame_graph import get_graph_geometries

logger = get_default_logger()


class PhonebotViewer(ProxyViewer):
    """
    Default viewer for phonebot.
    """

    def setup(self):
        super().setup()
        glvw = self.widget_

        # Setup reasonable camera position for phonebot
        glvw.setCameraPosition(elevation=30, azimuth=210, distance=0.5)

        # Add extra grid item
        grid_item = gl.GLGridItem()
        grid_item.setSpacing(0.02, 0.02, 0.02)
        glvw.addItem(grid_item)

        # Draw frames as axes
        pose_item = gl.GLLinePlotItem(mode='lines')
        self.items_['poses'] = pose_item
        self.handlers_['poses'] = self.draw_poses
        glvw.addItem(pose_item)

        # Draw poses
        edge_item = gl.GLLinePlotItem(mode='lines')
        self.items_['edges'] = edge_item
        self.handlers_['edges'] = self.draw_edges
        glvw.addItem(edge_item)

    def draw_poses(self, poses):
        lines = []
        colors = []
        for frame, pose in poses.items():
            lines.extend(primitives.principal_axes(pose, scale=0.01))
            colors.extend([[1, 0, 0], [1, 0, 0], [0, 1, 0],
                           [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)
        self.items_['poses'].setData(pos=lines, color=colors)

    def draw_edges(self, poses, edges):
        lines = []
        colors = []
        for edge in edges:
            if (edge.source in poses) and (edge.target in poses):
                source = poses[edge.source].position
                target = poses[edge.target].position
                lines.append([source, target])
                colors.extend([[1, 1, 1], [1, 1, 1]])
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)
        self.items_['edges'].setData(pos=lines, color=colors)
