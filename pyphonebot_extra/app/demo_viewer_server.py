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

from pyphonebot_extra.vis.viewer import PhonebotViewer
from pyphonebot_extra.vis.viewer.proxy_commands import AddPointsCommand, AddLinesCommand


class ViewerServer(object):
    def __init__(self, *args, **kwargs):
        self.data_queue = None
        self.event_queue = None
        self.command_queue = None
        self.server = SimpleServer(on_data=self.on_data, *args, **kwargs)

    def on_data(self, data: bytes):
        state = zlib.decompress(data)
        stamp, state = decode(state)
        graph = PhonebotGraph.decode(state)

        self.update(graph, stamp)

    def start(self):
        self.data_queue, self.event_queue, self.command_queue = PhonebotViewer.create()

    def run(self):
        return self.server.run()

    def update(self, graph: PhonebotGraph, stamp: float):
        cam_pos = graph.get_transform(
            FrameName.CAMERA, FrameName.LOCAL, stamp).position
        poses, edges = get_graph_geometries(
            graph, stamp, target_frame=FrameName.LOCAL, tol=np.inf)

        if not self.data_queue.full():
            visdata = {'poses': dict(poses=poses), 'edges':
                       dict(poses=poses, edges=edges)}
            self.data_queue.put_nowait(visdata)


def main():
    server = ViewerServer()
    server.start()
    server.run()


if __name__ == '__main__':
    main()
