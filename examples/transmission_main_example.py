#!/usr/bin/env python3
'''
Transmission setup example: 
reads from network json (default = examples/edfa/edfa_example_network.json)
propagates a 96 channels comb 
'''

from gnpy.core.equipment import equipment_from_json
from argparse import ArgumentParser
from sys import exit
from pathlib import Path
from json import loads
from collections import Counter
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG

from matplotlib.pyplot import show, axis, figure, title
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path)

from convert import convert_file
from gnpy.core import network_from_json, build_network
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.info import SpectralInformation, Channel, Power

logger = getLogger(__name__)

def plot_results(network, path, source, sink):
    path_edges = set(zip(path[:-1], path[1:]))
    edges = set(network.edges()) - path_edges
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
    draw_networkx_nodes(network, nodelist=path, node_color='#ff0000', **kwargs)
    draw_networkx_edges(network, edgelist=edges, edge_color='#ababab', **kwargs)
    draw_networkx_edges(network, edgelist=path_edges, edge_color='#ff0000', **kwargs)
    draw_networkx_labels(network, labels=labels, font_size=14, **{**kwargs, 'pos': label_pos})
    title(f'Propagating from {source.loc.city} to {sink.loc.city}')
    axis('off')
    show()


def load_network(filename, equipment):
    if args.filename.suffix.lower() == '.xls':
        logger.info('Automatically converting from XLS to gnpy JSON')
        json_data = convert_file(args.filename)
    else:
        with open(args.filename) as f:
            json_data = loads(f.read())
    return network_from_json(json_data, equipment)

def load_equipment(filename):
    with open(filename) as f:
        json_data = loads(f.read())
    return equipment_from_json(json_data, filename)

def main(network, equipment, source, sink):
    build_network(network, equipment=equipment)

    spacing = 0.05 # THz
    si = SpectralInformation() # SI units: W, Hz
    si = si.update(carriers=[
        Channel(f, (191.3 + spacing * f) * 1e12, 32e9, 0.15, Power(1e-3, 0, 0))
        for f in range(1,97)
    ])

    path = dijkstra_path(network, source, sink)
    logger.info(f'There are {len(path)} network elements between {source!r} and {sink!r}')

    logger.info('Propagating')
    for el in path:
        si = el(si)
        print(el)

    return path

parser = ArgumentParser()
parser.add_argument('-e', '--equipment', type=Path,
                    default=Path(__file__).parent / 'eqpt_config.json')
parser.add_argument('-p', '--plot', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-l', '--list-nodes', action='store_true', default=False, help='list all transceiver nodes')
parser.add_argument('filename', nargs='?', type=Path,
                    default=Path(__file__).parent / 'edfa_example_network.json')
parser.add_argument('source', nargs='?', help='source node')
parser.add_argument('sink',   nargs='?', help='sink node')

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={0: ERROR, 1: INFO, 2: DEBUG}.get(args.verbose, ERROR))

    equipment = load_equipment(args.equipment)
    # logger.info(equipment)

    network = load_network(args.filename, equipment)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}
    
    if not transceivers:
        exit('Network has no transceivers!')
    if len(transceivers) < 2:
        exit('Network has only one transceiver!')

    if args.list_nodes:
        for uid in transceivers:
            print(uid)
        exit()

    if args.source:
        try:
            source = next(transceivers[uid] for uid in transceivers if uid == args.source)
        except StopIteration as e:
            #TODO code a more advanced regex to find nodes match
            nodes_suggestion = [uid for uid in transceivers if args.source.lower() in uid.lower()]
            source = transceivers[nodes_suggestion[0]] if len(nodes_suggestion)>0 else transceivers[0]
            print(f'invalid souce node specified, did you mean:\
                  \n{nodes_suggestion}?\
                  \n{args.source!r}, replaced with {source.uid}')
            del transceivers[source.uid]
    else:
        logger.info('No source node specified: picking random transceiver')
        source = list(transceivers.values())[0]

    if args.sink:
        try:
            sink = next(transceivers[uid] for uid in transceivers if uid == args.sink)
        except StopIteration as e:
            nodes_suggestion = [uid for uid in transceivers if args.sink.lower() in uid.lower()]
            sink = transceiver[nodes_suggestion[0]] if len(nodes_suggestion)>0 else tansceivers[-1]
            print(f'invalid destination node specified, did you mean:\
                \n{nodes_suggestion}?\
                \n{args.sink!r}, replaced with {sink.uid}')
    else:
        logger.info('No source node specified: picking random transceiver')
        sink = list(transceivers.values())[-1]

    logger.info(f'source = {args.source!r}')
    logger.info(f'sink = {args.sink!r}')
    path = main(network, equipment, source, sink)

    if args.plot:
        plot_results(network, path, source, sink)
