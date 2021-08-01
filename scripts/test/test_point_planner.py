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
from phonebot.core.planning.cwp import CircleWorldPlanner


def sq_norm(x: np.ndarray):
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
        sqr_a = sq_norm(x - center_a)
        sqr_b = sq_norm(x - center_b)
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
    # seed = np.random.randint(2**16 - 1)
    seed = 53194
    print(F'seed={seed}')
    np.random.seed(seed)

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
        with timer('prm'):
            plan = prm.plan(q0, q1)
        with timer('cwp'):
            plan_2 = cwp.plan([q0, q1])
        # print(plan)
        # print(plan_2)

    with timer('draw'):
        x, y = ws.exterior.xy
        plt.gca().invert_yaxis()
        plt.grid()
        plt.plot(x, y, label='workspace')
        plt.plot(waypoints[..., 0], waypoints[..., 1], 'x-', label='waypoints')
        if plan is not None:
            plt.plot(plan[..., 0], plan[..., 1], '*--', label='prm')
        if plan_2 is not None:
            plt.plot(plan_2[..., 0], plan_2[..., 1], '*--', label='cwp')
        else:
            print(F'Failed to find plan: {plan}')
        plt.legend()

        # nx.draw(cwp.G)

        plt.show()


if __name__ == '__main__':
    main()
