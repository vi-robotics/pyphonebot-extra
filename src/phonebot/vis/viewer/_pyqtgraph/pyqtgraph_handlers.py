#!/usr/bin/env python3

__all__ = [
    'Plot2DHandler',
    'PointsHandler',
    'LinesHandler',
    'LineStripHandler',
    'FrameAxesHandler',
    'GridHandler',
    'PosesHandler',
    'EdgesHandler']

from typing import Tuple, Dict
import numpy as np
import logging

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.core.common.math.transform import Position, Rotation, Transform
from phonebot.vis.primitives import principal_axes
from phonebot.core.frame_graph.frame_edges import FrameEdge


class Plot2DHandler:
    """Example handler to deal with 2D plots.

    NOTE(ycho): Since all other handlers deal with 3D data,
    this is currently (ironically) the odd-one out.
    """

    def __init__(self, viewer: 'PyqtViewer3D'):
        # Create plot widget ...
        plw = pg.PlotWidget()
        viewer.layout_.addWidget(plw)

        # Create plot item ...
        item = pg.PlotItem()
        plw.addItem(item)

        data = pg.PlotDataItem()
        item.addItem(data)

        self.item = item
        self.data = data

    def __call__(self, *args, **kwargs):
        self.data.setData(*args, **kwargs)


class PointsHandler:
    """Add points."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        item = gl.GLScatterPlotItem()
        item.pos = np.empty((0, 3))  # prevent abort due to pyqtgraph bug
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, *args, **kwargs):
        self.item.setData(*args, **kwargs)


class LinesHandler:
    """Draw lines, as independent segments expressed as lists of point
    pairs."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        item = gl.GLLinePlotItem(mode='lines')
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, *args, **kwargs):
        self.item.setData(*args, **kwargs)


class LineStripHandler:
    """Draw lines, as continuous strips expressed as lists of point."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        self.viewer = viewer
        item = gl.GLLinePlotItem(mode='line_strip')
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, *args, **kwargs):
        return self.item.setData(*args, **kwargs)


class MeshHandler:
    """Draw 3D meshes."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        item = gl.GLMeshItem(meshdata=None)
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, *args, **kwargs):
        self.item.setMeshData(*args, **kwargs)


class FrameAxesHandler:
    """Add axes representing the principal vectors at a given pose."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        item = gl.GLLinePlotItem(mode='lines')
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, poses: Dict[str, Transform]):
        # 1. Build primitives.
        lines = []
        colors = []
        for frame, pose in poses.items():
            lines.extend(principal_axes(pose, scale=1.0))
            colors.extend([[1, 0, 0], [1, 0, 0], [0, 1, 0],
                           [0, 1, 0], [0, 0, 1], [0, 0, 1]])

        # 2. Convert to np.ndarray.
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)

        # 3. Update data.
        self.item.setData(pos=lines, color=colors)


class GridHandler:
    """Add a planar (z-plane) grid in the world."""

    def __init__(self, viewer: 'PyqtViewer3D',
                 size: Tuple[float, float, float] = (100, 100, 1),
                 spacing: Tuple[float, float, float] = (1, 1, 1)):
        self.size_ = size
        self.spacing_ = spacing

        item = gl.GLGridItem()
        item.setSize(*self.size_)
        item.setSpacing(*self.spacing_)
        viewer.widget_.addItem(item)
        self.item = item

    def __call__(self, size: Tuple[float, float, float] = None,
                 spacing: Tuple[float, float, float] = None):
        if size is not None:
            self.item.setSize(*size)
        if spacing is not None:
            self.item.setSpacing(*spacing)


class PosesHandler(LinesHandler):
    """Draw poses as rgb-colored frame axes."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        super().__init__(viewer)

    def __call__(self, poses):
        lines = []
        colors = []
        for frame, pose in poses.items():
            # FIXME(ycho): Hardcoded default scale...
            lines.extend(principal_axes(pose, scale=0.01))
            colors.extend([[1, 0, 0], [1, 0, 0], [0, 1, 0],
                           [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)
        return super().__call__(pos=lines, color=colors)


class EdgesHandler(LinesHandler):
    """Draw edges between poses as white lines."""

    def __init__(self, viewer: 'PyqtViewer3D'):
        super().__init__(viewer)

    def __call__(
            self, poses: Dict[str, Transform],
            edges: Tuple[FrameEdge, ...]):
        lines = []
        colors = []
        for edge in edges:
            if edge.source not in poses:
                logging.warn(F'Frame {edge.source} not found in `poses`')
                continue
            if edge.target not in poses:
                logging.warn(F'Frame {edge.target} not found in `poses`')
                continue
            source = poses[edge.source].position
            target = poses[edge.target].position
            lines.append([source, target])
            colors.extend([[1, 1, 1], [1, 1, 1]])
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)
        return super().__call__(pos=lines, color=colors)
