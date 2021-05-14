#!/usr/bin/env python3

from pathlib import Path
import time
import zlib
import numpy as np

from phonebot.core.common.math.utils import anorm, alerp
from phonebot.core.common.serial import encode, decode
from phonebot.core.common.config import PhonebotSettings, FrameName

from phonebot.core.frame_graph.phonebot_graph import PhonebotGraph
from phonebot.core.frame_graph.graph_utils import get_graph_geometries
from phonebot.core.common.comm.server import SimpleServer

from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer.viewer_base import HandleHelper


class ViewerServer(object):
    def __init__(self, *args, **kwargs):
        self.viewer = None
        self.handler = None
        self.server = SimpleServer(on_data=self.on_data, *args, **kwargs)

    def on_data(self, data: bytes):
        state = zlib.decompress(data)
        stamp, state = decode(state)
        graph = PhonebotGraph.decode(state)

        self.update(graph, stamp)

    def start(self):
        self.viewer = PhonebotViewer()
        self.handler = HandleHelper(self.viewer)

    def run(self):
        return self.server.run()

    def update(self, graph: PhonebotGraph, stamp: float):
        cam_pos = graph.get_transform(
            FrameName.CAMERA, FrameName.LOCAL, stamp).position
        poses, edges = get_graph_geometries(
            graph, stamp, target_frame=FrameName.LOCAL, tol=np.inf)

        with self.handler.collect():
            self.handler.poses(poses=poses)
            self.handler.edges(poses=poses, edges=edges)


def main():
    server = ViewerServer()
    server.start()
    server.run()


if __name__ == '__main__':
    main()
