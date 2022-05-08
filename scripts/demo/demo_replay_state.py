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

from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer.viewer_base import HandleHelper


def main():
    pattern = 'state-' + '[0-9]' * 4 + '.txt'
    files = list(Path('/tmp/phonebot-phone').glob(pattern))
    viewer = PhonebotViewer()
    handler = HandleHelper(viewer)

    count = 0
    for filename in sorted(files):
        count += 1
        if count < 120:
            continue
        # time.sleep(1.0)
        with open(filename, 'rb') as f:
            state = zlib.decompress(f.read())
            stamp, state = decode(state)
            print(stamp)
            graph = PhonebotGraph.decode(state)
        print(filename)

        # Send data to asynchronous viewer.
        # NOTE(yycho0108): All of the following code is concerned with
        # visualization.
        cam_pos = graph.get_transform(
            FrameName.CAMERA, FrameName.LOCAL, stamp).position
        poses, edges = get_graph_geometries(
            graph, stamp, target_frame=FrameName.LOCAL, tol=np.inf)
        with handler.collect():
            handler.poses(poses=poses)
            handler.edges(poses=poses, edges=edges)


if __name__ == '__main__':
    main()
