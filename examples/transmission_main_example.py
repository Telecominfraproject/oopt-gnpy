#!/usr/bin/env python3
'''
Transmission setup example: 
reads from network json (default = examples/edfa/edfa_example_network.json)
propagates a 96 channels comb 
'''

from gnpy.core.equipment import load_equipment
from gnpy.core.utils import db2lin, lin2db
from argparse import ArgumentParser
from sys import exit
from pathlib import Path
from json import loads
from collections import Counter
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG

from matplotlib.pyplot import show, axis, figure, title
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path)
from gnpy.core.network import load_network, build_network, set_roadm_loss, set_edfa_dp
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.info import create_input_spectral_information, Channel, Power, Pref

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


def main(network, equipment, source, sink):
    roadms = [roadm for roadm in network if isinstance(roadm, Roadm)]
    default_roadm_loss = equipment['Roadms']['default'].gain_mode_default_loss
    power_mode = equipment['Spans']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     f'=> it can be modified in eqpt_config.json - Spans']))
    set_roadm_loss(roadms, False, 0, default_roadm_loss)
    build_network(network, equipment=equipment)
    path = dijkstra_path(network, source, sink)
    if power_mode:
        path_amps = [amp for amp in path if isinstance(amp, Edfa)]
        set_edfa_dp(network, path_amps)     

    spans = [s.length for s in path if isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans):.0f}m between {source.uid} and {sink.uid}')
    print(f'\nNow propagating between {source.uid} and {sink.uid}:')

    pref_span_db = 0
    bounds = range(0, 1) #power sweep

    for p_db in range(pref_span_db+bounds.start, pref_span_db+bounds.stop): #change range to sweep results across several powers in dBm
        p = db2lin(p_db)*1e-3

        pref_roadm_db = equipment['Roadms']['default'].power_mode_pref #TODO parametrize in eqpt_json 
        roadm_loss = p_db - pref_roadm_db #dynamic update the ROADM loss wrto power sweep to keep the same pref_roadm        
        path_roadms = [roadm for roadm in path if isinstance(roadm, Roadm)]
        set_roadm_loss(path_roadms, power_mode, roadm_loss, default_roadm_loss)
            
        spacing = 0.05e12 
        bw = 32e9 #bandwidth Hz
        frequency_start = 191.3e12
        roll_off = 0.15
        nch = 96
        si = create_input_spectral_information(frequency_start, roll_off, bw, p, spacing, nch, p_db)

        print(f'\nPorpagating with input power = {lin2db(p*1e3):.2f}dBm :')
        for el in path:
            si = el(si)
            print(el) #remove this line when sweeping across several powers
        print(f'\nTransmission result for input power = {lin2db(p*1e3):.2f}dBm :')
        print(sink)

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
    print(args.filename)
    network = load_network(args.filename, equipment)
    print(network)

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
            nodes_suggestion = [uid for uid in transceivers \
                if args.source.lower() in uid.lower()]
            source = transceivers[nodes_suggestion[0]] \
                if len(nodes_suggestion)>0 else list(transceivers.values())[0]
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
            nodes_suggestion = [uid for uid in transceivers \
                if args.sink.lower() in uid.lower()]
            sink = transceivers[nodes_suggestion[0]] \
                if len(nodes_suggestion)>0 else list(transceivers.values())[0]
            print(f'invalid destination node specified, did you mean:\
                \n{nodes_suggestion}?\
                \n{args.sink!r}, replaced with {sink.uid}')
    else:
        logger.info('No source node specified: picking random transceiver')
        sink = list(transceivers.values())[1]

    logger.info(f'source = {args.source!r}')
    logger.info(f'sink = {args.sink!r}')
    path = main(network, equipment, source, sink)

    if args.plot:
        plot_results(network, path, source, sink)
