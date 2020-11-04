#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:     BayesianTracker
# Purpose:  A multi object tracking library, specifically used to reconstruct
#           tracks in crowded fields. Here we use a probabilistic network of
#           information to perform the trajectory linking. This method uses
#           positional and visual information for track linking.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  14/08/2014
# -------------------------------------------------------------------------------


__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"


import re
from itertools import combinations, product

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _draw_cube(ax, box):
    """ draw_cube

    Plots a cube of dimensions box into the current axes.

    Args:
        ax: a matplotlib3d axes object
        box: dimensions of the box e.g. [(0,1),(0,1),(0,1)]
    Notes:
        Note that the z-order of plotting is not correct.
        DONE(arl): make non cube (i.e. cuboid) shapes.
    """
    if not isinstance(ax, Axes):
        raise TypeError('ax argument must be a matpotlib axis')

    edges = combinations(np.array(list(product(box[0], box[1], box[2]))), 2)

    for s, e in edges:
        if any(
            [np.sum(np.abs(s - e)) == box[d][1] - box[d][0] for d in range(3)]
        ):
            ax.plot(*list(zip(s, e)), linestyle=':', color='b')


def _check_plot_order(order):
    """ check_plot_order

    Check that the plotting order is properly formatted. The plotting order is
    specified as a string (e.g. 'xyt') which denotes the axes to plot. Plots
    can either be of two (e.g. 'xz') or three dimensions (e.g. 'xyz').

    Args:
        order: a string containing the plotting order

    Returns:
        DIMS: the number of plotting dimensions if the plot order is properly
            specified, otherwise raises a ValueError
    """
    if not isinstance(order, str):
        raise ValueError('Order must be specified as a string, e.g. xyz')

    # check that the string contains plottable axes
    order_ok = not bool(re.compile(r'[^xyzt]').search(order))
    if not order_ok:
        raise ValueError(
            'Order string is incorrectly specified: {0:s}'
            '. Should contain one of xyzt.'.format(order)
        )

    # check that we have the correct number of dimensions
    DIMS = len(order)
    if DIMS < 2 or DIMS > 3:
        raise ValueError('Plot dimensions must be 2D or 3D.')

    return DIMS


def plot_tracks(
    tracks,
    agents=[],
    lw=1.0,
    terminii=False,
    tail=None,
    box=None,
    order='xyz',
    kalman=False,
    cmap=plt.get_cmap('viridis'),
    labels=False,
    title='BayesianTracker output',
):
    """ plot_tracks

    Plot tracks using matplotlib/matplotlib3d. Uses linecollections to speed up
    plotting of many tracks. Two and three dimensional plots can be achieved
    by specifying the order parameter, e.g. order='xy' or 'xyt'...

    Args:
        tracks: a list of tracklets
        order: plot order, default is xyz, but could be xyt etc..
        tail: limit the plotting to the last n timepoints
        box: plot a box to show the limits of the volume
        terminii: bool to plot markers at the start and end of tracks
        kalman: plot the output of the kalman filter
        labels: bool to plot track IDs as labels

    Returns:
        None

    Notes:
        DONE(arl): parse the order properly to elimate errors
        TODO(arl): implement limits on plotting
        TODO(arl): box plotting in 2D/3D

    """
    DIMS = _check_plot_order(order)

    fig = plt.figure(figsize=(16, 16))

    # set up fiddly plot functions
    if DIMS == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # use a color map
    colors_rgb = [cmap(int(i)) for i in np.linspace(0, 255, 16)]
    p_args = {
        'colors_rgb': colors_rgb,
        'order': order,
        'labels': labels,
        'lw': lw,
    }

    # render the tracks
    _render_tracks(ax, tracks, **p_args)

    # plot a box of the appropriate dimensions
    if box is not None and DIMS == 3:
        box_dims = 'xyzt'
        this_box = [box[box_dims.index(dim)] for dim in order]
        _draw_cube(ax, this_box)

    ax.set_xlabel(order[0])
    ax.set_ylabel(order[1])
    if DIMS == 3:
        ax.set_zlabel(order[2])
        # ax.set_zlim(0, 100)
    else:
        ax.autoscale('tight')

    # plt.axis('image')
    plt.title(title)
    plt.show()


def _render_tracks(
    ax,
    tracks,
    colors_rgb=[],
    order='xyz',
    labels=False,
    terminii=False,
    lw=1.0,
    use_alpha=False,
):
    """ render_tracks

    Render the tracks on a given set of axes using matplotlib/matplotlib3d.
    Uses linecollections to speed up plotting of many tracks. Two and three
    dimensional plots can be achieved by specifying the order parameter,
    e.g. order='xy' or 'xyt'...
    """
    DIMS = _check_plot_order(order)

    lines, clrs = [], []

    # set up fiddly plot functions
    if DIMS == 3:
        lineplot = lambda l, c: Line3DCollection(l, colors=c)
        addline = lambda ax, l: ax.add_collection3d(l)
    else:
        lineplot = lambda l, c: LineCollection(l, colors=c)
        addline = lambda ax, l: ax.add_collection(l)

    for track in tracks:

        if len(track) < 2:
            continue

        p_order = [getattr(track, order[i]) for i in range(DIMS)]
        segments = list(zip(*p_order))
        t_color = colors_rgb[track.ID % (len(colors_rgb) - 1)]

        lines.append(segments)
        clrs.append(t_color)

        # plot text labels on the tracks
        if labels:
            # set up the plotting arguments
            l_args = [getattr(track, order[i])[-1] for i in range(DIMS)]
            l_args = l_args + [str(track.ID), None]
            # plot the text label with an outline
            ax.text(
                *l_args,
                color='k',
                path_effects=[
                    PathEffects.withStroke(linewidth=0.5, foreground=t_color)
                ],
            )

        # TODO(arl): add the terminus plotting
        if terminii:
            pass

    lc = lineplot(lines, clrs)
    lc.set_linewidth(lw)

    addline(ax, lc)


if __name__ == "__main__":
    pass
