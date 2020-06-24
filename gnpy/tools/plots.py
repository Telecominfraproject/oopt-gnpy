#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.tools.plots
================

Graphs and plots usable form a CLI application
'''

from matplotlib.pyplot import show, axis, figure, title, text
from networkx import draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels
from gnpy.core.elements import Transceiver


def plot_baseline(network):
    edges = set(network.edges())
    pos = {n: (n.lng, n.lat) for n in network.nodes()}
    labels = {n: n.location.city for n in network.nodes() if isinstance(n, Transceiver)}
    city_labels = set(labels.values())
    for n in network.nodes():
        if n.location.city and n.location.city not in city_labels:
            labels[n] = n.location.city
            city_labels.add(n.location.city)
    label_pos = pos

    fig = figure()
    kwargs = {'figure': fig, 'pos': pos}
    plot = draw_networkx_nodes(network, nodelist=network.nodes(), node_color='#ababab', **kwargs)
    draw_networkx_edges(network, edgelist=edges, edge_color='#ababab', **kwargs)
    draw_networkx_labels(network, labels=labels, font_size=14, **{**kwargs, 'pos': label_pos})
    axis('off')
    show()


def plot_results(network, path, source, destination):
    path_edges = set(zip(path[:-1], path[1:]))
    edges = set(network.edges()) - path_edges
    pos = {n: (n.lng, n.lat) for n in network.nodes()}
    nodes = {}
    for k, (x, y) in pos.items():
        nodes.setdefault((round(x, 1), round(y, 1)), []).append(k)
    labels = {n: n.location.city for n in network.nodes() if isinstance(n, Transceiver)}
    city_labels = set(labels.values())
    for n in network.nodes():
        if n.location.city and n.location.city not in city_labels:
            labels[n] = n.location.city
            city_labels.add(n.location.city)
    label_pos = pos

    fig = figure()
    kwargs = {'figure': fig, 'pos': pos}
    all_nodes = [n for n in network.nodes() if n not in path]
    plot = draw_networkx_nodes(network, nodelist=all_nodes, node_color='#ababab', node_size=50, **kwargs)
    draw_networkx_nodes(network, nodelist=path, node_color='#ff0000', node_size=55, **kwargs)
    draw_networkx_edges(network, edgelist=edges, edge_color='#ababab', **kwargs)
    draw_networkx_edges(network, edgelist=path_edges, edge_color='#ff0000', **kwargs)
    draw_networkx_labels(network, labels=labels, font_size=14, **{**kwargs, 'pos': label_pos})
    title(f'Propagating from {source.loc.city} to {destination.loc.city}')
    axis('off')

    heading = 'Spectral Information\n\n'
    textbox = text(0.85, 0.20, heading, fontsize=14, fontname='Ubuntu Mono',
                   verticalalignment='top', transform=fig.axes[0].transAxes,
                   bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    msgs = {(x, y): heading + '\n\n'.join(str(n) for n in ns if n in path)
            for (x, y), ns in nodes.items()}

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
