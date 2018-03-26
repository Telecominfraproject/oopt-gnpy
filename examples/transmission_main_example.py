#!/usr/bin/env 
"""
@author: briantaylor
@author: giladgoldfarb
@author: jeanluc-auge

Transmission setup example: 
reads from network json (default = examples/edfa/edfa_example_network.json)
propagates a 96 channels comb 
"""
from argparse import ArgumentParser
from json import load
from sys import exit
from pathlib import Path
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG

from matplotlib.pyplot import show, axis, figure, title
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path)

from gnpy.core import network_from_json, build_network
from gnpy.core.elements import Transceiver, Fiber, Edfa
from gnpy.core.info import SpectralInformation, Channel, Power
#from gnpy.core.algorithms import closed_paths

logger = getLogger(__package__ or __file__)

def format_si(spectral_infos):
    return '\n'.join([
        f'#{idx} Carrier(frequency={c.frequency},\n  power=Power(signal={c.power.signal}, nli={c.power.nli}, ase={c.power.ase}))'
        for idx, si in sorted(set(spectral_infos))
        for c in set(si.carriers)
    ])

logger = getLogger('gnpy.core')

def main(args):
    with open(args.filename) as f:
        json_data = load(f)

    network = network_from_json(json_data)
    build_network(network)

    spacing = 0.05 #THz
    si = SpectralInformation() # !! SI units W, Hz
    si = si.update(carriers=tuple(Channel(f, (191.3+spacing*f)*1e12, 
            32e9, 0.15, Power(1e-3, 0, 0)) for f in range(1,97)))

    trx = [n for n in network.nodes() if isinstance(n, Transceiver)]
    if args.list>=1:
        print(*[el.uid[4:] for el in trx], sep='\n')
    else:
        try:
            source = next(el for el in trx if el.uid[4:] == args.source)
        except StopIteration as e:
            raise ValueError(f'invalid souce node specified: {args.source!r}') from e
        try:
            sink = next(el for el in trx if el.uid[4:] == args.sink)
        except StopIteration as e:
            raise ValueError(f'invalid souce node specified: {args.sink!r}') from e            

        path = dijkstra_path(network, source, sink)
        print(f'There are {len(path)} network elements between {source} and {sink}')

        for el in path:
            si = el(si)
            print(el)

        nodelist = [n for n in network.nodes() if isinstance(n, (Transceiver, Fiber))]
        pathnodes = [n for n in path if isinstance(n, (Transceiver, Fiber))]
        edgelist = [(u, v) for u, v in zip(pathnodes, pathnodes[1:])]
        node_color = ['#ff0000' if n is source or n is sink else
                      '#900000' if n in path else '#ffdfdf'
                      for n in nodelist]
        edge_color = ['#ff9090' if u in path and v in path else '#ababab'
                      for u, v in edgelist]
        labels = {n: n.location.city if isinstance(n, Transceiver) else ''
                  for n in pathnodes}

        fig = figure()
        pos = {n: (n.lng, n.lat) for n in nodelist}
        kwargs = {'figure': fig, 'pos': pos}
        plot = draw_networkx_nodes(network, nodelist=nodelist, node_color=node_color, **kwargs)
        draw_networkx_edges(network, edgelist=edgelist, edge_color=edge_color, **kwargs)
        draw_networkx_labels(network, labels=labels, font_size=14, **kwargs)
        title(f'Propagating from {source.loc.city} to {sink.loc.city}')
        axis('off')
        show()

parser = ArgumentParser()
parser.add_argument('filename', nargs='?', type=Path,
  default= Path(__file__).parent / 'edfa/edfa_example_network.json')
parser.add_argument('source', type=str, nargs='?', default="", help='source node')
parser.add_argument('sink', type=str, nargs='?', default="", help='sink node')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-l', '--list', action='count', default=0, help='list all network nodes')

if __name__ == '__main__':
    args = parser.parse_args()
    level = {1: INFO, 2: DEBUG}.get(args.verbose, ERROR)
    logger.setLevel(level)
    basicConfig()
    exit(main(args))
