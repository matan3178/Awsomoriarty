import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot2d(points_l1, points_l2, color1, color2):
    x1, x2 = [p[0] for p in points_l1], [p[0] for p in points_l2]
    y1, y2 = [p[1] for p in points_l1], [p[1] for p in points_l2]

    x = list()
    x.extend(x1)
    x.extend(x2)
    y = list()
    y.extend(y1)
    y.extend(y2)

    colors = [color1, color2]

    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()
    return


def plot3d(points_l1, points_l2, color1, color2):
    x1, x2 = [p[0] for p in points_l1], [p[0] for p in points_l2]
    y1, y2 = [p[1] for p in points_l1], [p[1] for p in points_l2]
    z1, z2 = [p[2] for p in points_l1], [p[2] for p in points_l2]

    x = list()
    x.extend(x1)
    x.extend(x2)
    y = list()
    y.extend(y1)
    y.extend(y2)
    z = list()
    z.extend(z1)
    z.extend(z2)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x1, y1, z1, c=color1, alpha=0.1)
    ax.scatter(x2, y2, z2, c=color2, alpha=0.1)
    plt.show()
    return