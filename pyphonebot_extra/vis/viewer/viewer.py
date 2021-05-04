#!/usr/bin/env python3

__all__ = ['Viewer', 'DummyViewer', 'PrimitiveViewer']

import time
import numpy as np
import functools
from abc import abstractmethod, ABC
import multiprocessing as mp
from multiprocessing import Process, Manager, Queue
import logging

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore


class EventWindow(pg.GraphicsWindow):
    """
    Simple window to forward UI events to signals.
    """
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


class GLViewWidget(gl.GLViewWidget):
    """
    Patched version of upstream GLViewWidget.
    """

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


class ViewerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def spin(self):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def on_key(self):
        raise NotImplementedError

    @abstractmethod
    def on_mouse(self):
        raise NotImplementedError


class Viewer(ViewerBase):
    """
    Generic asynchronous viewer.
    """

    def __init__(self, data_queue: Queue, event_queue: Queue):
        if QtGui.QApplication.instance() is None:
            # NOTE(ycho): Passing empty list instead of using sys.argv
            app = QtGui.QApplication([])
        else:
            app = QtGui.QApplication.instance()
        self.app_ = app
        self.win_ = EventWindow('Viewer')

        self.data_queue_ = data_queue
        self.event_queue_ = event_queue

        self.setup()

        # Boilerplate: connect render event every 50ms.
        self.timer_ = QtCore.QTimer()
        self.timer_.timeout.connect(self.on_render)
        self.timer_.start(50)

        # Handle key events
        self.win_.sigKeyPress.connect(self.on_key)
        self.win_.sigMousePress.connect(
            functools.partial(
                self.on_mouse,
                event_type='press'))
        self.win_.sigMouseReleased.connect(
            functools.partial(
                self.on_mouse,
                event_type='release'))
        self.win_.sigMouseMove.connect(
            functools.partial(
                self.on_mouse,
                event_type='move'))

    def spin(self):
        self.win_.show()
        return self.app_.exec_()

    def on_key(self, event):
        if not self.event_queue_.full():
            self.event_queue_.put_nowait(event.key())

    def on_mouse(self, event, event_type):
        if not self.event_queue_.full():
            self.event_queue_.put_nowait(
                (event_type, int(event.buttons()), event.x(), event.y()))

    def on_render(self):
        # TODO(yycho0108): enable render all
        # instead of just latest data?
        self.render(self.get_latest_data())

    def get_latest_data(self):
        data = None
        while not self.data_queue_.empty():
            data = self.data_queue_.get_nowait()
        return data

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def render(self, data):
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        def run_viewer(iq, oq, *args, **kwargs):
            viewer = cls(iq, oq, *args, **kwargs)
            return viewer.spin()
        iq = Queue(maxsize=128)
        oq = Queue(maxsize=128)
        p = Process(target=run_viewer,
                    args=(iq, oq) + args,
                    kwargs=kwargs,
                    daemon=True
                    )
        p.start()
        return iq, oq


class DummyViewer(Viewer):
    def setup(self):
        print('setup')

    def render(self, data):
        print('render : {}'.format(data))


class PrimitiveViewer(Viewer):
    def setup(self):
        self.layout_ = QtGui.QGridLayout()
        self.win_.setLayout(self.layout_)
        self.items_ = {}
        self.handlers_ = {}

        glvw = GLViewWidget(self.win_)

        # Line
        line_item = gl.GLLinePlotItem(mode='lines')
        self.items_['line'] = line_item
        self.handlers_['line'] = line_item.setData
        glvw.addItem(line_item)

        # Point
        # NOTE(ycho): initialize to np.ndarray since `list` invalid
        point_item = gl.GLScatterPlotItem(pos=np.empty((0, 3)))
        self.items_['point'] = point_item
        self.handlers_['point'] = point_item.setData
        glvw.addItem(point_item)

        # Mesh
        mesh_item = gl.GLMeshItem(meshdata=None)
        self.items_['mesh'] = mesh_item
        self.handlers_['mesh'] = mesh_item.setMeshData
        glvw.addItem(mesh_item)

        self.layout_.addWidget(glvw, 0, 0)
        glvw.sizeHint = lambda: pg.QtCore.QSize(100, 100)
        self.widget_ = glvw

    def render(self, data):
        if data is None:
            # No need to refresh panels
            return
        try:
            for name, kwargs in data.items():
                if name not in self.handlers_:
                    logging.warn("data tag {} not known".format(name))
                    continue
                self.handlers_[name](**kwargs)
        except Exception as e:
            logging.warn('Exception occurred during render : {}'.format(e))
            pass


def main():
    data_queue, event_queue = PrimitiveViewer.create()
    for _ in range(1000):
        if not data_queue.full():
            md = gl.MeshData.sphere(rows=4, cols=8)
            md.setVertexColors(md.vertexes())
            data_queue.put_nowait({
                'line': dict(pos=np.random.uniform(size=(5, 2, 3))),
                'point': dict(pos=np.random.uniform(size=(32, 3))),
                'mesh': dict(meshdata=md)
            })
        time.sleep(0.2)


if __name__ == '__main__':
    main()
