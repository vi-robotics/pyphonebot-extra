#!/usr/bin/env python3

from abc import abstractmethod, ABC
from enum import Enum
from typing import Tuple, Dict, List, Callable, Any, Hashable
from contextlib import contextmanager


class ViewerKey(Enum):
    MOUSE = 0
    KEY = 1
    REGISTER = 2
    UNREGISTER = 3
    RENDER = 4


class ViewerState(Enum):
    NULL = 0
    OPEN = 1
    CLOSED = 2


class ViewerBase(ABC):
    """Abstract viewer base.

    - render (client -> viewer) : Process data with handler
    - register (client -> viewer) : Add data handler
    - event (viewer -> client) : Forward Key/Mouse Events
    - state (viewer -> client) : Check viewer status
    """

    def __init__(self):
        pass

    @abstractmethod
    def register(self, name: Hashable, handler: Callable):
        """add data handler."""
        raise NotImplementedError

    @abstractmethod
    def unregister(self, name: Hashable):
        """add data handler."""
        raise NotImplementedError

    @abstractmethod
    def handle(self, data: Dict[Hashable, Tuple[Tuple, Dict]]):
        """data -> viewer."""
        raise NotImplementedError

    @abstractmethod
    def on_event(self, topic: str, callback: Callable) -> int:
        """viewer -> event_cb(key)"""
        raise NotImplementedError

    @abstractmethod
    def on_key(self, key_cb) -> int:
        """viewer -> key_cb(key)"""
        raise NotImplementedError

    @abstractmethod
    def on_mouse(self, mouse_cb) -> int:
        """viewer -> mouse_cb(key)"""
        raise NotImplementedError

    @abstractmethod
    def start(self) -> int:
        """Start the viewer."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> int:
        """Stop a started viewer."""
        raise NotImplementedError

    @abstractmethod
    def state(self) -> ViewerState:
        """Stop a started viewer."""
        raise NotImplementedError


@contextmanager
def with_attr(obj, name, value):
    """Temporarily set object attributes."""
    replace = hasattr(obj, name)

    if replace:
        try:
            old_value = getattr(obj, name)
        except AttributeError as e:
            raise

    setattr(obj, name, value)
    try:
        yield obj
    finally:
        if replace:
            setattr(obj, name, old_value)


class HandleHelper:
    """Helper class for using `Viewer` without ugly syntax.

    Refer to the apps/ directory in `extras` for sample usages.
    """

    def __init__(self, viewer: ViewerBase):
        self.viewer = viewer
        self.eager: bool = True
        self.data = {}

    def __handle(self, data):
        res = self.viewer.handle(data)
        self.data = {}
        return res

    def __getitem__(self, key: Hashable) -> Callable:
        def _handle(*args, **kwds):
            if self.eager:
                return self.__handle({key: (args, kwds)})
            else:
                self.data[key] = (args, kwds)
        return _handle

    def __getattr__(self, key: Hashable) -> Callable:
        return self.__getitem__(key)

    @contextmanager
    def collect(self, eager: bool = False):
        try:
            yield with_attr(self, 'eager', eager)
        finally:
            self.__handle(self.data)
