#!/usr/bin/env python3

import numpy as np
import time
from phonebot.vis.viewer import ProxyViewer
from phonebot.vis.viewer.proxy_command import ProxyCommand
from phonebot.vis.viewer.proxy_commands import AddPlotCommand


def main():
    data_queue, event_queue, command_queue = ProxyViewer.create()
    command_queue.put_nowait(AddPlotCommand(name='plot'))

    while True:
        x = np.linspace(-np.pi, np.pi)
        y = np.random.normal(size=len(x))
        if not data_queue.full():
            data_queue.put_nowait(dict(
                plot=dict(x=x, y=y)))
        time.sleep(0.001)


if __name__ == '__main__':
    main()
