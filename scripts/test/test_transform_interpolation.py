#!/usr/bin/env python3

import numpy as np
import time

import pyqtgraph.opengl as gl

from phonebot.core.frame_graph import FrameGraph
from phonebot.core.frame_graph.frame_edges import SimpleFrameEdge, StaticFrameEdge
from phonebot.core.common.math.transform import Transform, Rotation
from phonebot.core.common.math.utils import tlerp, normalize, anorm
from phonebot.core.common.math.utils import tlerp_geodesic

from phonebot.vis.viewer.proxy_command import ProxyCommand
from phonebot.vis.viewer.proxy_commands import AddLineStripCommand, AddLinesCommand, AddAxesCommand
from phonebot.vis.viewer import ProxyViewer


def main():
    num_points = 128

    source = Transform.random()
    target = Transform.random()

    data_queue, _, command_queue = ProxyViewer.create()
    command_queue.put(AddAxesCommand('poses'))
    command_queue.put(AddLinesCommand('edges'))
    command_queue.put(AddLineStripCommand('path'))

    path = np.zeros(shape=(num_points, 3))
    weights = np.linspace(0.0, 1.0, num=num_points)

    graph = FrameGraph()
    graph.add_edge(StaticFrameEdge('source', 'origin', source))
    graph.add_edge(StaticFrameEdge('target', 'origin', target))

    graph.add_edge(SimpleFrameEdge('current', 'origin'))
    graph.add_edge(SimpleFrameEdge('current', 'source'))
    graph.add_edge(SimpleFrameEdge('target', 'current'))

    stamp = time.time()
    for index, weight in enumerate(weights):
        # current = tlerp(source, target, weight)
        current = tlerp_geodesic(source, target, weight)
        graph.get_edge('current', 'origin').update(stamp, current)

        poses = {}
        for frame in graph.frames:
            poses[frame] = graph.get_transform(frame, 'origin', stamp)

        epos = []
        for edge in graph.edges:
            v0 = poses[edge.source].position
            v1 = poses[edge.target].position
            epos.append([v0, v1])
        epos = np.reshape(epos, (-1, 3))

        path[index] = current.position
        if not data_queue.full():
            data_queue.put_nowait({
                'poses': dict(poses=poses),
                'edges': dict(pos=epos, color=(1, 1, 1, 1)),
                'path': dict(pos=path[:index+1], color=(1, 1, 0, 1))
            })
        time.sleep(0.1)

    # print(' Compare ... ')
    # print(np.real(sqrtm(alpha_from_beta.to_matrix())))
    # print(alpha_from_gamma.to_matrix())


if __name__ == '__main__':
    main()
