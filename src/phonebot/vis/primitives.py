#!/usr/bin/env python3

from phonebot.core.common.math.transform import Position, Rotation, Transform


def line(source: Position, target: Position):
    return [source, target]


def principal_axes(transform: Transform, scale=1.0):
    o = transform * Position([0, 0, 0])
    x = transform * Position([scale, 0, 0])
    y = transform * Position([0, scale, 0])
    z = transform * Position([0, 0, scale])
    return line(o, x), line(o, y), line(o, z)


def ellipsoid(pose, dimensions):
    pass


def sphere(pose, radius):
    pass


def cuboid(pose, dimensions):
    pass


def cube(pose, radius):
    pass
