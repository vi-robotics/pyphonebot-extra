#!/usr/bin/env python3

__all__ = ['AddPlotCommand', 'AddLineStripCommand',
           'AddAxesCommand', 'AddLinesCommand']

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyphonebot_extra.vis import primitives
from pyphonebot_extra.vis.viewer.proxy_viewer import ProxyViewer
from pyphonebot_extra.vis.viewer.proxy_command import ProxyCommand


class AddPlotCommand(ProxyCommand):
    """
    Example command to add plot handler to proxy viewer.
    """

    def __init__(self, name: str = 'plot'):
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        plw = pg.PlotWidget()
        viewer.layout_.addWidget(plw)
        item = pg.PlotItem()
        plw.addItem(item)
        if self.name in viewer.items_:
            logger.warn('Name {} already registered'.format(self.name))
        viewer.items_[self.name] = item
        viewer.handlers_[self.name] = item.plot


class AddLineStripCommand(ProxyCommand):
    """
    Add a line strip, a polyline going through all specified points.
    """

    def __init__(self, name='line_strip'):
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLLinePlotItem(mode='line_strip')
        viewer.items_[self.name] = item
        viewer.handlers_[self.name] = item.setData
        viewer.widget_.addItem(item)


class AddAxesCommand(ProxyCommand):
    """
    Add axes representing the principal vectors at a given pose.
    """

    def __init__(self, name='axes'):
        super().__init__(name)

    @staticmethod
    def on_data(item, poses):
        lines = []
        colors = []
        for frame, pose in poses.items():
            lines.extend(primitives.principal_axes(pose, scale=1.0))
            colors.extend([[1, 0, 0], [1, 0, 0], [0, 1, 0],
                           [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        lines = np.stack(lines, axis=0)
        colors = np.stack(colors, axis=0)
        item.setData(pos=lines, color=colors)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLLinePlotItem(mode='lines')
        viewer.items_[self.name] = item
        viewer.handlers_[
            self.name] = lambda poses: AddAxesCommand.on_data(item, poses)
        viewer.widget_.addItem(item)


class AddLinesCommand(ProxyCommand):
    """
    Add lines, drawn as independent segments expressed as point pairs.
    """

    def __init__(self, name='lines'):
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLLinePlotItem(mode='lines')
        viewer.items_[self.name] = item
        viewer.handlers_[self.name] = item.setData
        viewer.widget_.addItem(item)


class AddGridCommand(ProxyCommand):
    """
    Add a planar (z-plane) grid in the world.
    """

    def __init__(self, name='grid', size=(100, 100, 1), spacing=(1, 1, 1)):
        self.size_ = size
        self.spacing_ = spacing
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLGridItem()
        item.setSize(*self.size_)
        item.setSpacing(*self.spacing_)
        viewer.items_[self.name] = item
        viewer.widget_.addItem(item)


class AddPointsCommand(ProxyCommand):
    """
    Add points.
    """

    def __init__(self, name='points'):
        super().__init__(name)

    def __call__(self, viewer: ProxyViewer):
        item = gl.GLScatterPlotItem()
        item.pos = np.empty((0, 3))  # prevent abort due to pyqtgraph bug
        viewer.items_[self.name] = item
        viewer.handlers_[self.name] = item.setData
        viewer.widget_.addItem(item)
