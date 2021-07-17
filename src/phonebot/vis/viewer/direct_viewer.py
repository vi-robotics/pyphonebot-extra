#!/usr/bin/env python3

import logging
import re
from typing import Tuple, Dict, List, Callable, Any, Hashable

from phonebot.vis.viewer.viewer_base import ViewerBase, ViewerState


class DirectViewer(ViewerBase):
    """Backend-agnostic synchronous viewer."""

    def __init__(self):
        self.handlers_ = {}
        self.event_cbs_: Dict[str, List] = {'key': [], 'mouse': []}
        self.state_: ViewerState = ViewerState.NULL

    def register(self, name: Hashable,
                 handler_fn: Callable[[ViewerBase], Any]):
        """Register event handler.

        Args:
            name: Event handler identifier; must be a valid key type.
            handler_fn: Function to instantiate handler with the viewer as argument.
        Returns:
            `True` if registration was successful; otherwise `False`.
        """
        if name in self.handlers_:
            logging.warn('Existing handler : {}'.format(self.handlers_[name]))
            return False
        self.handlers_[name] = handler_fn(self)
        return True

    def unregister(self, name: Hashable):
        """Unregister event handler.

        Args:
            name: Event handler identifier; must be a valid key type.
        """
        if name in self.handlers_:
            del self.handlers_[name]

    def handle(self, data: Dict[Hashable, Tuple[Tuple, Dict]]):
        """Upon receiving data, callback corresponding handlers.

        Args:
            data: dictionary of handler identifiers and callback arguments.
        """
        if data is None:
            return
        for name, (args, kwds) in data.items():
            if name not in self.handlers_:
                logging.warn('No such handler : {}'.format(name))
                continue
            # Render with data ...
            self.handlers_[name](*args, **kwds)

    def trigger(self, topic: str, data):
        """Trigger an event synchronously.

        NOTE(ycho): In general, handle() invokes server(=viewer)-side processors for
        client-side data, whereas trigger() invokes client-side processors for server-side data.
        FIXME(ycho): This symmetry is not well-reflected in the name, but for now it's fine
        to consider trigger() as the reversed-directional analogue of handle().

        Args:
            topic: The event topic; e.g. "key" or "mouse"
            data: The event payload.
        """
        for pattern, cbs in self.event_cbs_.items():
            if re.match(pattern, topic) is None:
                continue
            done = False
            for event_cb in cbs:
                # TODO(ycho): if event_cb returns false, no further processing is
                # done on this same data. This behavior is not documented, and I'm not
                # quite sure if it would be useful to leave it this way.
                if event_cb(topic, data):
                    done = True
            if done:
                break

    def on_event(self, topic: str, callback: Callable[[str, Any], bool]):
        """Register callbacks for events.

        Args:
            topic: The event topic; e.g. "key" or "mouse"
            callback: The callback function that accepts the expected payload.
        """
        # FIXME(ycho): `defaultdict`?
        if topic not in self.event_cbs_:
            self.event_cbs_[topic] = []
        self.event_cbs_[topic].append(callback)

    def on_key(self, key_cb):
        """Keyboard event registration."""
        self.on_event('key', lambda topic, data: key_cb(data))

    def on_mouse(self, mouse_cb):
        """Mouse event registration."""
        self.on_event('mouse', lambda topic, data: mouse_cb(data))

    def start(self):
        """Start the viewer.

        TODO(ycho): In general, `ViewerState` management is done poorly throughout the viewer stack.
        """
        self.state_ = ViewerState.OPEN

    def stop(self):
        """Stop the viewer.

        TODO(ycho): In general, `ViewerState` management is done poorly throughout the viewer stack.
        """
        self.state_ = ViewerState.CLOSED

    def state(self):
        """Retrieve the current viewer state.

        TODO(ycho): In general, `ViewerState` management is done poorly throughout the viewer stack.
        """
        return self.state_
