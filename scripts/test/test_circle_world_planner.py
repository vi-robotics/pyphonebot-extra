#!/usr/bin/env python3

import numpy as np
from tqdm.auto import tqdm

from phonebot.core.common.config import PhonebotSettings
from phonebot.core.common.math.utils import normalize, norm, adiff
from phonebot.core.planning.cwp import CircleWorldPlanner
from phonebot.core.common.geometry.geometry import (
    circle_point_intersects, circle_segment_intersects)


def _sq_norm(x: np.ndarray):
    """x^T@x; when in doubt, use einsum."""
    return np.einsum('...i,...i->...', x, x)


def generate_non_overlapping_circles(n: int):
    out = np.zeros((n, 3), dtype=np.float32)
    index = 0
    while index < n:
        circle = np.random.normal(size=3)
        circle[2] = np.abs(circle[2])

        #if index < 2:
        #    circle[2] = 1e-8

        if (_sq_norm(out[:index, :2] - circle[None, :2])
                > np.square(circle[2] + out[:index, 2])).all():
            out[index] = circle
            index += 1
    return out


def main():
    # NOTE(ycho): Set RNG seed.
    seed = np.random.randint(2**16 - 1)
    print(F'seed:{seed}')
    np.random.seed(seed)

    circles = generate_non_overlapping_circles(64)
    cwp = CircleWorldPlanner(circles)

    # Generate obstacle-free waypoints.
    best_path = []
    for _ in tqdm(range(128)):
        while True:
            min_bound = np.min(circles[..., :2] - circles[..., 2:], axis=0)
            max_bound = np.max(circles[..., :2] + circles[..., 2:], axis=0)
            wpts = np.random.uniform(
                low=min_bound, high=max_bound,
                size=(2, 2))
            if not circle_point_intersects(
                    circles[:, None],
                    wpts[None, :]).any():
                break
        path = cwp.plan(wpts)
        if len(path) > len(best_path):
            best_path = path
            best_wpts = wpts
    path = best_path
    wpts = best_wpts

    if True:
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Arc
        from matplotlib.colors import TABLEAU_COLORS

        # draw world
        for c in circles:
            p = plt.Circle((c[0], c[1]), radius=c[2], fill=False, color='c')
            plt.gca().add_patch(p)
        plt.gca().set_aspect('equal', adjustable='datalim')

        # draw plan
        segments = []
        for e in cwp.G.edges(data=True):
            ed = e[2]

            p0 = cwp.G.nodes[e[0]]['pos']
            p1 = cwp.G.nodes[e[1]]['pos']
            ps = np.stack([p0, p1], axis=0)

            if ed['center'] is None:
                # straight line
                # plt.plot(ps[..., 0], ps[..., 1], 'k*--', alpha=0.5)
                segments.append(ps)
            else:
                continue
                # arc (hugging edge)
                c = circles[ed['center']]
                if not np.isfinite(c).all():
                    continue
                h0 = np.arctan2(*(p0 - c[:2])[::-1])
                h1 = np.arctan2(*(p1 - c[:2])[::-1])
                if not np.isfinite([h0, h1]).all():
                    continue
                dh = adiff(h1, h0)
                href = h1 if dh < 0 else h0

                hmax = np.maximum(0.0, np.rad2deg(np.abs(dh)) - 2.0)
                hmin = np.minimum(2.0, hmax)

                arc = Arc(
                    c[: 2],
                    2 * c[2],
                    2 * c[2],
                    np.rad2deg(href),
                    hmin, hmax,
                    edgecolor=list(TABLEAU_COLORS.values())
                    [np.random.choice(len(TABLEAU_COLORS))],
                    linewidth=8, linestyle='--')
                plt.gca().add_patch(arc)
                # plt.plot([x0, x1], [y0, y1], 'r*--', alpha=0.5)
        col = LineCollection(segments, color='gray', alpha=0.5, linestyle='--')
        plt.gca().add_collection(col)
        plt.plot(path[..., 0], path[..., 1], 'r-')
        plt.plot(wpts[..., 0], wpts[..., 1], 'ro', markersize=10)
        # plt.plot(*path[-1], 'bo', markersize=10)
        plt.gca().plot()
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
