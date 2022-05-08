#!/usr/bin/env python3

__all__ = ['PyqtApplication', 'PyqtViewer']

from typing import Tuple, Dict, List, Callable, Any
import functools
import logging
import time
from functools import partial
import re

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.vis.viewer.direct_viewer import DirectViewer
from phonebot.vis.viewer.viewer_base import ViewerBase, ViewerKey


class EventWindow(pg.GraphicsWindow):
    """Simple window to forward UI events to signals."""
    sigKeyPress = QtCore.pyqtSignal(object)
    sigMousePress = QtCore.pyqtSignal(object)
    sigMouseMove = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableMouse()

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)
        super().keyPressEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.sigMouseReleased.emit(ev)
        super().mouseReleaseEvent(ev)
        ev.ignore()

    def mouseMoveEvent(self, ev):
        self.sigMouseMove.emit(ev)
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        self.sigMousePress.emit(ev)
        super().mousePressEvent(ev)


class PyqtApplication:
    """GUI Appllication with PyQt[5].

    Creates a PyQt application with a standard interface regarding event
    propagation (via Qt signals) and start/stop mechanism. Tries to be
    general when possible.
    """

    def __init__(
            self, on_key: Callable[[Tuple[int]], Any],
            on_mouse: Callable[[Tuple[str, int, int, int]], Any]):
        """Create Interactive Pyqt Application with GUI events.

        Args:
            on_key: Keyboard event callback, called as on_key(key_id)
            on_mouse: Mouse event callback, called as on_mouse(event_type,button,x,y)
                where `event_type is one of ('press', 'release', 'mouse').
                and `button` is an integer-valued button state (see Qt enums).
        """
        super().__init__()

        # Save callbacks.
        self.on_key_ = on_key
        self.on_mouse_ = on_mouse

        # Gui Setup
        app = QtGui.QApplication.instance()
        if QtGui.QApplication.instance() is None:
            # NOTE(ycho): Passing empty list instead of using sys.argv
            app = QtGui.QApplication([])

        self.app_ = app
        self.win_ = EventWindow('Viewer')

        # Subscribe to internal Qt events
        # and translate these events to our format.
        self.win_.sigKeyPress.connect(self.__on_key)
        self.win_.sigMousePress.connect(
            functools.partial(
                self.__on_mouse,
                event_type='press'))
        self.win_.sigMouseReleased.connect(
            functools.partial(
                self.__on_mouse,
                event_type='release'))
        self.win_.sigMouseMove.connect(
            functools.partial(
                self.__on_mouse,
                event_type='move'))

    def __on_key(self, event: QtGui.QKeyEvent):
        """Internal keyboard event handler."""
        data = (event.key(),)
        self.on_key_(data)

    def __on_mouse(self, event: QtGui.QMouseEvent, event_type: str):
        """Internal mouse event handler."""
        data = (event_type, int(event.buttons()),
                event.x(), event.y())
        self.on_mouse_(data)

    def start(self):
        """Start viewer and block execution."""
        self.win_.show()

        # NOTE(ycho): Some Qt issues prevent
        # handling of KeyboardInterrupt() with exec_().
        # Thus, we explicitly run the event loop ourselves.
        # The performance hit is (hopefully) not that great.
        # return self.app_.exec_()
        while True:
            self.app_.processEvents()

    def stop(self):
        """Stop viewer."""
        if self.app_ is not None:
            self.app_.quit()


class PyqtViewer(DirectViewer):
    """Direct synchronous viewer."""

    class Signals(QtCore.QObject):
        """Signals to forward function calls to main thread."""
        register = QtCore.pyqtSignal(tuple, dict)
        unregister = QtCore.pyqtSignal(tuple, dict)
        handle = QtCore.pyqtSignal(tuple, dict)

    def __init__(self):
        super().__init__()

        # NOTE(ycho): Start app and connect with self ui events.
        on_key = partial(self.trigger, 'key')
        on_mouse = partial(self.trigger, 'mouse')
        self.app_ = PyqtApplication(on_key, on_mouse)

        # All public-facing calls are forwarded through
        # qt signal-slot mechanism to ensure running on the Gui thread.
        self.signals = PyqtViewer.Signals()
        self.signals.register.connect(
            lambda args, kwds: DirectViewer.register(self, *args, **kwds))
        self.signals.unregister.connect(
            lambda args, kwds: DirectViewer.unregister(self, *args, **kwds))
        self.signals.handle.connect(
            lambda args, kwds: DirectViewer.handle(self, *args, **kwds))

    def register(self, *args, **kwds):
        """Override default behavior; call via Qt signals."""
        self.signals.register.emit(args, kwds)

    def unregister(self, *args, **kwds):
        """Override default behavior; call via Qt signals."""
        self.signals.unregister.emit(args, kwds)

    def handle(self, *args, **kwds):
        """Override default behavior; call via Qt signals."""
        self.signals.handle.emit(args, kwds)

    def start(self):
        """Start viewer."""
        res = super().start()
        self.app_.start()
        return res

    def stop(self):
        """Stop viewer."""
        self.app_.stop()
        return super().stop()
