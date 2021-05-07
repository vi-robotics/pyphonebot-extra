#!/usr/bin/env python3

import numpy as np
from phonebot.core.common.geometry.geometry import circle_intersection

from matplotlib import pyplot as plt

def plot_circle(x,y,l):
    h = np.linspace(-np.pi, np.pi)
    c,s = np.cos(h), np.sin(h)

    px = x + c * l
    py = y + s * l
    plt.plot(px, py, '-')

def main():
    x0, y0, l0 = 0.011, 0.0, 0.0175
    x1, y1, l1 = 0.0031390283667031907, 0.006682395882637915, 0.0285

    res1, res2 = circle_intersection((x0, y0, l0), (x1, y1, l1))
    print(res1, res2)
    plot_circle(x0,y0,l0)
    plot_circle(x1,y1,l1)
    plt.plot([res1[0]], [res2[1]], '+')
    plt.plot([res1[0]], [res2[1]], '+')
    plt.legend()

    plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
