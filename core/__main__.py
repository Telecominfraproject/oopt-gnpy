#!/usr/bin/env python3

from argparse import ArgumentParser
from json import load
from sys import exit
from pathlib import Path
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
from random import choice
from itertools import islice, cycle, groupby

from matplotlib.pyplot import show, axis, title, figure, annotate
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels)

from . import network_from_json
from .elements import Transceiver, Fiber
from .info import SpectralInformation, Channel, Power
from .algorithms import closed_paths

logger = getLogger(__package__ or __file__)

def format_si(spectral_infos):
    return '\n'.join([
        f'#{idx} Carrier(frequency={c.frequency},\n  power=Power(signal={c.power.signal}, nli={c.power.nli}, ase={c.power.ase}))'
        for idx, si in sorted(set(spectral_infos))
        for c in set(si.carriers)
    ])

def format_path(path):
    return ' → \n'.join(
        ' → '.join(f'{n.loc.city}' for n in group if isinstance(n, Transceiver))
        for _, group in groupby(path, key=lambda _, c=cycle('a'*30+'b'*30): next(c)))

def main(args):
    with open(args.filename) as f:
        json_data = load(f)

    network = network_from_json(json_data)
    pos    = {n: (n.long, n.lat) for n in network.nodes()}
    labels_pos = {n: (long-.5, lat-.5) for n, (long, lat) in pos.items()}
    size   = [20 if isinstance(n, Fiber) else 80 for n in network.nodes()]
    labels = {n: n.location.city if isinstance(n, Transceiver) else ''
              for n in network.nodes()}

    si = SpectralInformation(
        Channel(1, 193.95e12, '16-qam', 32e9, 0,  # 193.95 THz, 32 Gbaud
            Power(1e-3, 1e-6, 1e-6)),             # 1 mW, 1uW, 1uW
        Channel(1, 195.95e12, '16-qam', 32e9, 0,  # 195.95 THz, 32 Gbaud
            Power(1.2e-3, 1e-6, 1e-6)),           # 1.2 mW, 1uW, 1uW
    )

    nodes = [n for n in network.nodes() if isinstance(n, Transceiver)]
    source, sink = choice(nodes), choice(nodes)

    results = list(islice(closed_paths(network, source, sink, si), 3))
    paths = [[n for _, n, _ in r] for r in results]
    infos = {}
    for idx, r in enumerate(results):
        for in_si, node, out_si in r:
            infos.setdefault(node, []).append((idx, out_si))

    node_color = ['#ff0000' if n is source or n is sink else
                  '#900000' if any(n in p for p in paths) else
                  '#ffdede' if isinstance(n, Transceiver) else '#dedeff'
                  for n in network.nodes()]
    edge_color = ['#ff9090' if any(u in p for p in paths) and
                               any(v in p for p in paths) else '#dedede'
                  for u, v in network.edges()]

    fig = figure()
    plot = draw_networkx_nodes(network, pos=pos, node_size=size, node_color=node_color, figure=fig)
    draw_networkx_edges(network, pos=pos, figure=fig, edge_color=edge_color)
    draw_networkx_labels(network, pos=labels_pos, labels=labels, font_size=14, figure=fig)

    title(f'Propagating from {source.loc.city} to {sink.loc.city}')
    axis('off')
    tooltip = annotate('', xy=(0, 0), xytext=(20, 20),textcoords='offset points',
                       size=18, arrowprops={'arrowstyle': '->'},
                       bbox={'alpha': .7, 'fc': 'red'})
    tooltip.set_visible(False)

    fig.axes[0].text(-.1, -.1, '\n\n'.join(f'#{idx} {format_path(p)}'
                                           for idx, p in enumerate(paths)),
                     transform=fig.axes[0].transAxes,
                     bbox={'facecolor': 'red', 'alpha': .5, 'pad': 10})

    def hover(event):
        if event.inaxes not in fig.axes:
            return
        contained, indices = plot.contains(event)
        if not contained:
            tooltip.set_visible(False)
        elif not tooltip.get_visible():
            idx = indices['ind'][0]
            node = list(network.nodes())[idx]
            if not any(node in p for p in paths):
                return
            text = format_si(infos[node])
            tooltip.xy = plot.get_offsets()[idx]
            tooltip.set_text(text)
            tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    show()

parser = ArgumentParser()
parser.add_argument('filename', nargs='?', type=Path,
  default= Path(__file__).parent / '../examples/coronet.conus.json')
parser.add_argument('-v', '--verbose', action='count')

if __name__ == '__main__':
    args = parser.parse_args()
    level = {1: INFO, 2: DEBUG}.get(args.verbose, ERROR)
    logger.setLevel(level)
    basicConfig()
    exit(main(args))
