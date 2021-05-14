#!/usr/bin/env python3

import time
import numpy as np
import functools
from abc import abstractmethod, ABC
import multiprocessing as mp
from tqdm import tqdm
import threading
import logging
from typing import Tuple, Dict, List, Callable, Any

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.core.common.queue_listener import QueueListener

from phonebot.vis.viewer.viewer_base import ViewerState
from phonebot.vis.viewer.viewer_base import HandleHelper
from phonebot.vis.viewer._pyqtgraph import PyqtViewer3D
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_handlers import LineStripHandler
from phonebot.vis.viewer.async_viewer import AsyncViewer


class KeyCameraHandler:
    def __init__(self, viewer: PyqtViewer3D):
        self.viewer = viewer

    def __call__(self, dy, dz):
        self.viewer.widget_.orbit(dz, dy)


def pack(*args, **kwds):
    """Utility for grouping (args,kwds)"""
    return (args, kwds)


def main():
    logging.basicConfig(level=logging.INFO)
    use_async = True
    if use_async:
        viewer = AsyncViewer(PyqtViewer3D)
    else:
        viewer = PyqtViewer3D()
    handler = HandleHelper(viewer)
    viewer.register('camera', KeyCameraHandler)
    viewer.register('line_strip', LineStripHandler)

    def on_mouse(data):
        print('on_mouse --------- data = {}'.format(data))

    def on_key(data):
        c = data[0]
        if c in [ord(x) for x in 'WASD']:
            c = chr(c).lower()
            dz, dy = 0, 0
            if c == 'w':
                dy = +5
            if c == 's':
                dy = -5
            if c == 'a':
                dz = -5
            if c == 'd':
                dz = +5
            handler.camera(dy=dy, dz=dz)

        if c == ord('Q'):
            viewer.stop()

    viewer.on_mouse(on_mouse)
    viewer.on_key(on_key)

    def draw_stuff():
        for _ in tqdm(range(1024)):
            if viewer.state() == ViewerState.CLOSED:
                break
            md = gl.MeshData.sphere(rows=4, cols=8)
            md.setVertexColors(md.vertexes())
            with handler.collect():
                handler.line(pos=np.random.uniform(size=(1, 2, 3)))
                handler.point(pos=np.random.uniform(size=(32, 3)))
                handler.mesh(meshdata=md)
                handler.line_strip(pos=np.random.uniform(size=(32, 3)))
            time.sleep(0.01)

    try:
        viewer.start()
    except KeyboardInterrupt:
        viewer.stop()

    t = threading.Thread(target=draw_stuff)
    t.start()

    try:
        t.join()
    finally:
        viewer.stop()


if __name__ == '__main__':
    main()
