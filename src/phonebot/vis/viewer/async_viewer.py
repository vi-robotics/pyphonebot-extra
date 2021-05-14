#!/usr/bin/env python3

import threading
import multiprocessing as mp
import re
from typing import Tuple, Dict, List, Callable, Any, Hashable

from phonebot.core.common.queue_listener import QueueListener

from phonebot.vis.viewer.viewer_base import (
    ViewerBase, ViewerState, ViewerKey)


class AsyncClient:
    """Async communication viewer client that communicates with the server (in a different process) over mp.Queue."""

    def __init__(self,
                 cls: Callable[..., ViewerBase],
                 data_queue: mp.Queue, event_queue: mp.Queue,
                 args: List, kwds: Dict):
        """

        Create async client.

        TODO(ycho): If possible, consider replacing cls with viewer_fn to rid of the
        ugly syntax with `args`/`kwds` arguments passed around everywhere.

        Args:
            cls: The synchronous viewer class.
            data_queue: Data queue for viewer to handle; client->server(viewer)
            event_queue: Event queue for client to handle; server(viewer)->client
            args: Positional args for instantiation, i.e. cls(*args,**kwds)
            kwds: Keyword args for instantiation, i.e. cls(*args,**kwds)

        """
        super().__init__()
        self.data_queue = data_queue
        self.event_queue = event_queue
        self.viewer = cls(*args, **kwds)
        # NOTE(ycho): Total forwarding with regex `.*?`.
        self.viewer.on_event('.*?', self.__on_event)

        self.data_listener = QueueListener(
            self.data_queue,
            self.__on_data, daemon=False)

    def __on_data(self, packet):
        dtype, data = packet
        # FIXME(ycho): RENDER -> HANDLE
        if dtype == ViewerKey.RENDER:
            self.handle(data)
        elif dtype == ViewerKey.UNREGISTER:
            name = data
            self.unregister(name)
        elif dtype == ViewerKey.REGISTER:
            name, handler = data
            self.register(name, handler)

    def __on_event(self, topic: str, data):
        """ Internal event handler """
        self.event_queue.put_nowait((topic, data))
        # By default, don't take ownership of event
        return False

    def register(self, name: str, handler: Callable):
        """ Delegate handler registration """
        return self.viewer.register(name, handler)

    def unregister(self, name: str):
        """ Delegate handler un-registration """
        return self.viewer.unregister(name)

    def handle(self, data):
        """ Delegate handle() """
        return self.viewer.handle(data)

    def start(self):
        """ Delegate start() but also with communication listeners """
        self.data_listener.start()
        self.viewer.start()

    def stop(self):
        """ Delegate stop() but also with communication listeners """
        if (self.data_listener is not None and not self.data_listener.stopped()):
            self.data_listener.stop()
            if threading.current_thread() != self.data_listener:
                self.data_listener.join()
            self.data_listener = None
        self.viewer.stop()

    def state(self) -> ViewerState:
        """ Delegate state() """
        return self.viewer.state()

    def __del__(self):
        """ Stop first and delete self """
        self.stop()
        super().__del__()

    @classmethod
    def run(cls, *args, **kwds):
        """ Convenience method for creating and starting an instance. """
        instance = cls(*args, **kwds)
        return instance.start()


class AsyncViewer(ViewerBase):
    """Wrapper class for asynchronous operation."""

    def __init__(
            self,
            viewer_fn: Callable[[Any], ViewerBase],
            start: bool = True,
            *args, **kwds):
        """Create Async Viewer.

        Args:
            viewer_fn: Function to create a viewer instance.
            start: Whether to start the viewer immediately (in the constructor).
            *args: Positional args for instantiation, i.e. viewer_fn(*args,**kwds)
            **kwds: Keyword args for instantiation, i.e. viewer_fn(*args,**kwds)
        """
        super().__init__()

        # Create communication handles
        # NOTE(ycho): `event` : viewer(server) -> client
        # NOTE(ycho): `data`  : client -> viewer(server)
        self.event_cbs_: Dict[str, List] = {'key': [], 'mouse': []}

        # Save queues
        self.event_queue: mp.Queue = mp.Queue()
        self.data_queue: mp.Queue = mp.Queue()

        # Start viewer client.
        p = mp.Process(
            target=AsyncClient.run,
            args=(
                viewer_fn,
                self.data_queue,
                self.event_queue,
                args,
                kwds), daemon=True)
        self.process = p

        # Start event listener.
        self.event_listener = QueueListener(
            self.event_queue,
            lambda event: self.__on_event(event[0], event[1]),
            daemon=False)

        # Depending on the init arg, start immediately.
        if start:
            self.start()

    def __on_event(self, topic: str, data):
        """internal event handler."""
        for key, cbs in self.event_cbs_.items():
            if re.match(key, topic) is None:
                continue
            done = False
            for event_cb in cbs:
                if event_cb(topic, data):
                    done = True
            if done:
                break

    def register(self, name: Hashable, handler: Callable):
        """Register handler."""
        self.data_queue.put_nowait((ViewerKey.REGISTER, (name, handler)))

    def unregister(self, name: Hashable):
        """Un-register handler."""
        self.data_queue.put_nowait((ViewerKey.UNREGISTER, name))

    def handle(self, data):
        """Handle data."""
        self.data_queue.put_nowait((ViewerKey.RENDER, data))

    def on_key(self, key_cb):
        """interface keyboard event registration."""
        self.on_event('key', lambda topic, data: key_cb(data))

    def on_mouse(self, mouse_cb):
        """interface mouse event registration."""
        self.on_event('mouse', lambda topic, data: mouse_cb(data))

    def on_event(self, topic: str, event_cb):
        """interface general event registration."""
        # FIXME(ycho): `defaultdict`?
        if topic not in self.event_cbs_:
            self.event_cbs_[topic] = []
        self.event_cbs_[topic].append(event_cb)

    def stop(self):
        """Stop async viewer."""
        if (self.event_listener is not None and not self.event_listener.stopped()):
            self.event_listener.stop()
            if threading.current_thread() != self.event_listener:
                self.event_listener.join()
            self.event_listener = None
        self.process.terminate()
        self.process.join()

    def start(self):
        """Start async viewer."""
        if self.state() == ViewerState.OPEN:
            return
        self.process.start()
        self.event_listener.start()

    def state(self) -> ViewerState:
        """Get async viewer running state"""
        return ViewerState.OPEN if self.process.is_alive() else ViewerState.CLOSED

    def __del__(self):
        self.stop()
