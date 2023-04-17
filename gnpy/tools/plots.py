#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.tools.plots
================

Graphs and plots usable from a CLI application
"""

from matplotlib.pyplot import show, axis, figure, title, text
from networkx import draw_networkx
from gnpy.core.elements import Transceiver


def _try_city(node):
    return node.location.city if node.location.city else node.uid


def plot_baseline(network):
    pos = {n: (n.lng, n.lat) for n in network.nodes()}
    labels = {n: _try_city(n) for n in network.nodes() if isinstance(n, Transceiver)}
    draw_networkx(network, pos=pos, node_size=50, node_color='#ababab', edge_color='#ababab',
                  labels=labels, font_size=14)
    axis('off')
    show()


def plot_results(network, path, source, destination):
    path_edges = set(zip(path[:-1], path[1:]))
    edges = set(network.edges()) - path_edges
    nodes = [n for n in network.nodes() if n not in path]
    pos = {n: (n.lng, n.lat) for n in network.nodes()}
    nodes_by_pos = {}
    for k, (x, y) in pos.items():
        nodes_by_pos.setdefault((round(x, 1), round(y, 1)), []).append(k)

    labels = {n: _try_city(n) for n in network.nodes() if isinstance(n, Transceiver)}

    fig = figure()
    draw_networkx(network, pos=pos, labels=labels, font_size=14,
                  nodelist=nodes, node_color='#ababab', node_size=50,
                  edgelist=edges, edge_color='#ababab')
    draw_networkx(network, pos=pos, with_labels=False,
                  nodelist=path, node_color='#ff0000', node_size=55,
                  edgelist=path_edges, edge_color='#ff0000')
    title(f'Propagating from {_try_city(source)} to {_try_city(destination)}')
    axis('off')

    heading = 'Spectral Information\n\n'
    textbox = text(0.85, 0.20, heading, fontsize=14, fontname='Ubuntu Mono',
                   verticalalignment='top', transform=fig.axes[0].transAxes,
                   bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    msgs = {(x, y): heading + '\n\n'.join(str(n) for n in ns if n in path)
            for (x, y), ns in nodes_by_pos.items()}

    def hover(event):
        if event.xdata is None or event.ydata is None:
            return
        if fig.contains(event):
            x, y = round(event.xdata, 1), round(event.ydata, 1)
            if (x, y) in msgs:
                textbox.set_text(msgs[x, y])
            else:
                textbox.set_text(heading)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    show()
