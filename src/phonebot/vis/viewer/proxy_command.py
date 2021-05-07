#!/usr/bin/env python3

__all__ = ['ProxyCommand']

from abc import ABC, abstractmethod
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from phonebot.vis.viewer import Viewer


class ProxyCommand(ABC):
    """
    Class defining a simple interface to ProxyViewer.

    NOTE(yycho0108): the convention used here is that
    the classes intended to be used as a proxy command should be implemented
    as a functor (function object), such that the object can be pickeld.
    """

    def __init__(self, name: str = ''):
        self.name_ = name

    @property
    def name(self):
        return self.name_

    @abstractmethod
    def __call__(self, viewer: Viewer):
        return NotImplemented
