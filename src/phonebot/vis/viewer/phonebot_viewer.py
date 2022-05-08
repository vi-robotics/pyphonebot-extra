#!/usr/bin/env python3

import numpy as np

from phonebot.vis.viewer.viewer_base import (
    ViewerBase, ViewerState, ViewerKey)
from phonebot.vis.viewer._pyqtgraph import PyqtViewer3D
from phonebot.vis.viewer._pyqtgraph.pyqtgraph_handlers import (
    GridHandler, PosesHandler, EdgesHandler)
from phonebot.vis.viewer.async_viewer import AsyncViewer
from phonebot.vis.primitives import principal_axes


class PhonebotViewer(AsyncViewer):
    """AsyncViewer with some default handlers configured."""

    def __init__(self, base_viewer: ViewerBase = PyqtViewer3D):
        """Instantiate Phonebot Viewer with reasonable default handlers.

        Args:
            base_viewer: Synchronous viewer server based for AsyncViewer.
                @see AsyncViewer.
        """
        super().__init__(base_viewer)
        self.register('grid', GridHandler)
        self.register('poses', PosesHandler)
        self.register('edges', EdgesHandler)
