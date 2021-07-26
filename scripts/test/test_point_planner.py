#!/usr/bin/env python3

import numpy as np
from functools import partial
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
from contextlib import contextmanager
from typing import Tuple
import itertools
import networkx as nx
from collections import defaultdict

from phonebot.core.common.math.utils import normalize, norm
from phonebot.core.common.config import PhonebotSettings
from phonebot.core.kinematics.workspace import get_workspace
from phonebot.core.planning.prm import PRM


class CircleWorldPlanner:
    """Optimal shortest-path planner in a circular world.

    Reference:
    https://redblobgames.github.io/circular-obstacle-pathfinding/
    """

    def __init__(self, circles: Tuple[Tuple[float, float, float], ...]):
        self.circles = np.asarray(circles, dtype=np.float32)
        self.edges = []
        self._generate_graph()

    def _generate_graph(self):
        # Compute index pairs.
        n = len(self.circles)
        indices = itertools.combinations(range(n), 2)  # nC2,2
        i0, i1 = np.transpose(list(indices))

        # Inverse map: circle index
        cimap = defaultdict(list)
        for ii, (i0_, i1_) in enumerate(zip(i0, i1)):
            cimap[i0_].append(ii)  # src
            cimap[i1_].append(-ii)  # dst

        G = nx.Graph()

        c0 = self.circles[i0]
        c1 = self.circles[i1]
        r0 = c0[..., 2]
        r1 = c1[..., 2]
        dp = c1[..., :2] - c0[..., :2]
        d = norm(dp)
        u = dp / d
        ux, uy = u[..., 0], u[..., 1]
        R = np.asarray([[ux, -uy], [uy, ux]], dtype=np.float32)

        # Internal bitangents
        c = (r0 + r1) / d.squeeze(axis=-1)
        s = np.sqrt(1 - np.square(c))
        c, s = np.einsum('abn, bn -> an', R, [c, s])

        src, dst = [], []
        src.append(c0[..., :2] + r0 * np.stack([c, s], axis=-1))
        src.append(c0[..., :2] + r0 * np.stack([c, -s], axis=-1))
        dst.append(c1[..., :2] + r1 * np.stack([-c, s], axis=-1))
        dst.append(c1[..., :2] + r1 * np.stack([-c, -s], axis=-1))

        # External bitangents
        c = (r0 - r1) / d.squeeze(axis=-1)
        s = np.sqrt(1 - np.square(c))
        c, s = np.einsum('abn, bn -> an', R, [c, s])
        src.append(c0[..., :2] + r0 * np.stack([c, s], axis=-1))
        src.append(c0[..., :2] + r0 * np.stack([c, -s], axis=-1))
        dst.append(c1[..., :2] + r1 * np.stack([c, s], axis=-1))
        dst.append(c1[..., :2] + r1 * np.stack([c, -s], axis=-1))

        # Finalize list of surfing edges
        src = np.concatenate(src, axis=0)
        dst = np.concatenate(dst, axis=0)
        dist = norm(src - dst).squeeze(axis=-1)
        m = len(dist)
        for i, e in enumerate(dist):
            # NOTE(ycho): nodes and edges are indexed by combinatoric order.
            G.add_node(i, pos=src[i])
            G.add_node(-i, pos=dst[i])
            G.add_edge(i, -i, length=e)

        # Add list of hugging edges to the graph.
        # p = src[i0==i] | dst[i1==i]
        def _node_pos(i: int):
            if i < 0:
                return dst[-i]
            else:
                return src[i]

        for i in range(n):
            # Circle index -> edge indices that touch said circle
            circle = self.circles[i]
            center = circle[:2]
            radius = circle[2]
            sqr = np.square(radius)

            node_indices = cimap[i]

            for i0, i1 in itertools.combinations(node_indices, 2):
                p0 = _node_pos(i0)
                p1 = _node_pos(i1)
                cos_theta = (p0 - center).dot(p1 - center) / sqr
                arclen = radius * np.arccos(cos_theta)
                G.add_edge(i0, i1, length=arclen)

        self.G = G

    @classmethod
    def from_phonebot(cls, cfg: PhonebotSettings):
        small_radius = cfg.knee_link_length - cfg.hip_link_length
        sqr0 = np.square(small_radius)
        circles = (
            (cfg.hip_joint_offset, 0, small_radius),
            (-cfg.hip_joint_offset, 0, small_radius)
        )
        return cls(circles)


def _sq_norm(x: np.ndarray):
    """When in doubt, use einsum."""
    return np.einsum('...i,...i->...', x, x)


def _get_sample_fn(workspace: Polygon):
    # Figure out workspace bounds ...
    x, y = workspace.exterior.xy
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(x), np.max(x)

    def sample(shape: int):
        # Sanitize `shape` arg ...
        if isinstance(shape, int):
            shape = (shape,)
        shape = shape + (2,)

        # Feed through uniform distribution.
        return np.random.uniform([xmin, ymin], [xmax, ymax],
                                 shape)
    return sample


def _get_query_fn(workspace: Polygon):
    def query(x: np.ndarray):
        res = [workspace.contains(Point(*e)) for e in x.reshape(-1, 2)]
        return ~np.asarray(res).reshape(x.shape[:-1])
    return query


def _fast_query_fn(cfg: PhonebotSettings):
    small_radius = cfg.knee_link_length - cfg.hip_link_length
    large_radius = cfg.knee_link_length + cfg.hip_link_length

    sqr0 = np.square(small_radius)
    sqr1 = np.square(large_radius)
    center_a = (cfg.hip_joint_offset, 0)
    center_b = (-cfg.hip_joint_offset, 0)

    def query(x: np.ndarray):
        sqr_a = _sq_norm(x - center_a)
        sqr_b = _sq_norm(x - center_b)
        return ~np.logical_and.reduce([
            sqr0 <= sqr_a,
            sqr0 <= sqr_b,
            sqr_a < sqr1,
            sqr_b < sqr1])
    return query


def _get_plan_fn(query_fn, N: int = 128):
    def plan(x0, x1):
        x = np.linspace(x0, x1, N, False)
        return ~query_fn(x).any(axis=0)
    return plan


@contextmanager
def timer(name: str):
    """Simple timer."""
    import time
    try:
        t0 = time.time()
        yield
    finally:
        t1 = time.time()
        dt = 1000.0 * (t1 - t0)
        print(F'{name} took {dt} ms')


def main():
    np.random.seed(21)

    with timer('init'):
        cfg = PhonebotSettings()
        ws = get_workspace(0, cfg, True)

        # Sampler
        sample_fn = _get_sample_fn(ws)
        # query_fn = _get_query_fn(ws)
        query_fn = _fast_query_fn(cfg)
        plan_fn = _get_plan_fn(query_fn)
        prm = PRM(sample_fn, query_fn, plan_fn)

        cwp = CircleWorldPlanner.from_phonebot(cfg)

    with timer('construct'):
        prm.construct()

    with timer('waypoint'):
        # Construct start-goal positions ...
        waypoints = []
        while len(waypoints) < 2:
            s = sample_fn(())
            if ws.contains(Point(s)):
                waypoints.append(s)
        waypoints = np.asarray(waypoints)

    with timer('plan'):
        q0 = waypoints[0]
        q1 = waypoints[1]
        plan = prm.plan(q0, q1)

    with timer('draw'):
        x, y = ws.exterior.xy
        plt.gca().invert_yaxis()
        plt.grid()
        plt.plot(x, y)
        plt.plot(waypoints[..., 0], waypoints[..., 1], 'x-')
        if plan is not None:
            plt.plot(plan[..., 0], plan[..., 1], '*--')
        else:
            print(F'Failed to find plan: {plan}')

        nx.draw(cwp.G)

        plt.show()


if __name__ == '__main__':
    main()
