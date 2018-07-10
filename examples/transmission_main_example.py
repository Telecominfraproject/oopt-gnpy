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
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power, Pref
from gnpy.core.request import Path_request, RequestParams, compute_constrained_path, propagate

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


def main(network, equipment, source, sink, req = None):
    build_network(network, equipment=equipment)
    
    path = compute_constrained_path(network,req)
    spans = [s.length for s in path if isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans):.0f}m between {source.uid} and {sink.uid}')
    print(f'\nNow propagating between {source.uid} and {sink.uid}:')
    
    for p in range(0, 1): #change range to sweep results across several powers in dBm
        req.power = db2lin(p)*1e-3
        print(f'\nPropagating with input power = {lin2db(req.power*1e3):.2f}dBm :')
        propagate(path,req,equipment,show=True)
        print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f}dBm :')

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
    # print(network)

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

    params = {}
    params['request_id'] = 0
    params['source'] = source.uid
    params['destination'] = sink.uid
    params['trx_type'] = 'vendorA_trx-type1'
    params['trx_mode'] = 'PS_SP64_1'
    params['nodes_list'] = [sink.uid]
    params['loose_list'] = ['strict']
    params['spacing'] = 50e9
    params['power'] = 0
    params['nb_channel'] = 97
    params['frequency'] = equipment['Transceiver'][params['trx_type']].frequency
    try:
        extra_params = next(m 
            for m in equipment['Transceiver'][params['trx_type']].mode 
                if  m['format'] == params['trx_mode'])
    except StopIteration :
        msg = f'could not find tsp : {params} with mode: {params} in eqpt library'
        raise ValueError(msg)
    params.update(extra_params)
    req = Path_request(**params)
    path = main(network, equipment, source, sink,req)

    if args.plot:
        plot_results(network, path, source, sink)
