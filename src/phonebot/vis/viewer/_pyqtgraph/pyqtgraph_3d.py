#!/usr/bin/env python3

__all__ = ['PyqtViewer3D']

from typing import Tuple, Dict, List, Callable, Any
import functools
import logging
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.vis.viewer.viewer_base import ViewerBase, ViewerKey
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_backend import PyqtViewer, EventWindow
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_handlers import (
    LinesHandler, PointsHandler, MeshHandler)


class GLViewWidget(gl.GLViewWidget):
    """Patched version of upstream GLViewWidget; actually fires events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mousePos = None

    def mouseMoveEvent(self, ev):
        if self.mousePos is None:
            self.mousePos = ev.pos()
        if isinstance(self.parent(), EventWindow):
            self.parent().sigMouseMove.emit(ev)
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        if isinstance(self.parent(), EventWindow):
            self.parent().sigMousePress.emit(ev)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if isinstance(self.parent(), EventWindow):
            self.parent().sigMouseReleased.emit(ev)
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if isinstance(self.parent(), EventWindow):
            self.parent().sigKeyPress.emit(ev)
        super().keyPressEvent(ev)


class PyqtViewer3D(PyqtViewer):
    """PyqtViewer that has been initially setup with 3D OpenGL Viewer."""

    def __init__(self):
        super().__init__()
        app = self.app_

        # Setup GUI layout.
        self.layout_ = QtGui.QGridLayout()
        app.win_.setLayout(self.layout_)
        self.items_ = {}

        # Instantiate OpenGL 3D Widget by default...
        glvw = GLViewWidget(app.win_)
        # NOTE(ycho): Set reasonable looking defaults
        # FIXME(ycho): Hardcoded camera parameters!
        glvw.setCameraPosition(elevation=30, azimuth=210, distance=0.5)

        # Add to layout
        glvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.layout_.addWidget(glvw, 0, 0)
        self.widget_ = glvw

        # Setup reasonable defaults for handlers on commonly used primitives.
        self.register('line', LinesHandler)
        self.register('point', PointsHandler)
        self.register('mesh', MeshHandler)
