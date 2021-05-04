#!/usr/bin/env python3

from pyphonebot_extra.vis import primitives
from phonebot.core.frame_graph import FrameGraph
from phonebot.core.frame_graph import get_graph_geometries

#from mayavi import mlab
# def draw_line(line, *args, **kwargs):
#    source, target = line
#    mlab.plot3d([source.x, target.x], [source.y, target.y],
#                [source.z, target.z], *args, **kwargs)


def draw_principal_axes(axes, *args, **kwargs):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for axis, color in zip(axes, colors):
        draw_line(axis, *args, color=color, **kwargs)


def draw_graph(graph: FrameGraph, stamp: float, tol: float = 1e-1):
    poses, edges = get_graph_geometries(graph, stamp, tol)

    for frame, pose in poses.items():
        axes = primitives.principal_axes(pose, scale=0.01)
        draw_principal_axes(axes, tube_radius=0.002)

    for edge in edges:
        if (edge.source in poses) and (edge.target in poses):
            source = poses[edge.source].position
            target = poses[edge.target].position
            draw_line([source, target], color=(0, 0, 0), tube_radius=0.001)


def draw_polygon(ax, polygon):
    """
    Draw a polygon.
    """
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)


def draw_leg(js, ax, prefix,
             point='o', line='-',
             **kwargs):
    """
    Render single leg in its coordinate frames.
    """
    ja0, ja1, ja2, jb0, jb1, jb2 = js
    jvals = dict(
        ja0=ja0, ja1=ja1, ja2=ja2,
        jb0=jb0, jb1=jb1, jb2=jb2
    )
    handles = []
    for k in 'ja0,ja1,ja2,jb0,jb1,jb2'.split(','):
        v = jvals[k]
        handles.extend(
            ax.plot([v[0]], [v[1]], point, label='{}{}'.format(prefix, k), **kwargs))

    for k1, k2 in [('ja0', 'ja1'), ('ja1', 'ja2'), ('jb0', 'jb1'), ('jb1', 'jb2')]:
        v1 = jvals[k1]
        v2 = jvals[k2]
        handles.extend(
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], line, **kwargs))
    return handles
