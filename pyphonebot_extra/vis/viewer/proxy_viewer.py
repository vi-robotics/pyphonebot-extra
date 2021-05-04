#!/usr/bin/env python3

__all__ = ['ProxyViewer']

import logging
import time
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.core.common.logger import get_default_logger
from phonebot.core.common.queue_listener import QueueListener
from pyphonebot_extra.vis.viewer import Viewer, PrimitiveViewer
from pyphonebot_extra.vis.viewer.proxy_command import ProxyCommand

from threading import Thread
from multiprocessing import Process, Manager, Queue


logger = get_default_logger()


class ProxyCommandHandler(QtCore.QObject):
    signal = QtCore.pyqtSignal(object)

    def __init__(self, parent):
        super().__init__()
        self.parent_ = parent
        self.signal.connect(self.on_command)

    def on_command(self, command: ProxyCommand):
        # TODO(yycho0108): consider limiting the scope of command,
        # i.e. pass widget/app/ ... instead of `self`.
        # On the other hand, this scheme maximizes flexibility.
        logger.info('Received command : {}'.format(command))
        command(self.parent_)


class ProxyViewer(PrimitiveViewer):
    """
    Thin wrapper layer to forward commands to remote process.
    """

    def __init__(self, data_queue, event_queue, command_queue):
        super().__init__(data_queue, event_queue)
        self.command_queue_ = command_queue
        self.command_handler_ = ProxyCommandHandler(self)
        self.command_listener_ = QueueListener(
            command_queue, self.command_handler_.signal.emit)
        self.command_listener_.start()

    @classmethod
    def create(cls, *args, **kwargs):
        command_queue = Queue(maxsize=128)
        data_queue, event_queue = super().create(command_queue)
        return data_queue, event_queue, command_queue
