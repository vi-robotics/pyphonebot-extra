#!/usr/bin/env python3

import numpy as np
import time
from phonebot.vis.viewer.viewer_base import HandleHelper
from phonebot.vis.viewer.phonebot_viewer import PhonebotViewer
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_handlers import Plot2DHandler



def main():
    viewer = PhonebotViewer()
    viewer.register('plot', Plot2DHandler)
    viewer.start()
    handler = HandleHelper(viewer)

    while True:
        x = np.linspace(-np.pi, np.pi)
        y = np.random.normal(size=len(x))
        handler.plot(x=x, y=y)
        time.sleep(0.01)
    viewer.stop()


if __name__ == '__main__':
    main()
