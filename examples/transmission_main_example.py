#!/usr/bin/env python3
'''
Transmission setup example: 
reads from network json (default = examples/edfa/edfa_example_network.json)
propagates a 96 channels comb 
'''

from gnpy.core.equipment import load_equipment, trx_mode_params
from gnpy.core.utils import db2lin, lin2db
from argparse import ArgumentParser
from sys import exit
from pathlib import Path
from json import loads
from collections import Counter
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
from numpy import arange
from matplotlib.pyplot import show, axis, figure, title
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path)
from gnpy.core.network import load_network, build_network, set_roadm_loss, set_edfa_dp
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power, Pref
from gnpy.core.request import Path_request, RequestParams, compute_constrained_path, propagate

logger = getLogger(__name__)

def plot_results(network, path, source, destination):
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
    title(f'Propagating from {source.loc.city} to {destination.loc.city}')
    axis('off')
    show()


def main(network, equipment, source, destination, req = None):
    power_mode = equipment['Spans']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     f'=> it can be modified in eqpt_config.json - Spans']))    

    set_roadm_loss(network, equipment, False, 0)
    build_network(network, equipment=equipment)
    path = compute_constrained_path(network, req)
    if power_mode: 
        set_edfa_dp(network, path)

    spans = [s.length for s in path if isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans):.0f}m between {source.uid} and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')

    pref_span_db = lin2db(req.power*1e3)
    try:
        power_range = arange(*equipment['SI']['default'].power_range_db)
    except TypeError:
        print('invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')
        power_range = [0]

    for dp_db in power_range:
        p_db = pref_span_db + dp_db
        pref_roadm_db = equipment['Roadms']['default'].power_mode_pref
        roadm_loss = p_db - pref_roadm_db #dynamic update the ROADM loss wrto power sweep to keep the same pref_roadm
        set_roadm_loss(network, equipment, power_mode, roadm_loss)        
        p = db2lin(p_db)*1e-3
        req.power = p
        print(f'\nPropagating with input power = {lin2db(req.power*1e3):.2f}dBm :')
        propagate(path, req, equipment, show=len(power_range)==1)
        print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f}dBm :')        
        print(destination)
    return path


parser = ArgumentParser()
parser.add_argument('-e', '--equipment', type=Path,
                    default=Path(__file__).parent / 'eqpt_config.json')
parser.add_argument('-pl', '--plot', action='store_true', default=False)
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-l', '--list-nodes', action='store_true', default=False, help='list all transceiver nodes')
parser.add_argument('-po', '--power', default=0, help='channel ref power in dBm')
#parser.add_argument('-plb', '--power-lower-bound', default=0, help='power sweep lower bound')
#parser.add_argument('-pub', '--power-upper-bound', default=1, help='power sweep upper bound')
parser.add_argument('filename', nargs='?', type=Path,
                    default=Path(__file__).parent / 'edfa_example_network.json')
parser.add_argument('source', nargs='?', help='source node')
parser.add_argument('destination',   nargs='?', help='destination node')


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

    if args.destination:
        try:
            destination = next(transceivers[uid] for uid in transceivers if uid == args.destination)
        except StopIteration as e:
            nodes_suggestion = [uid for uid in transceivers \
                if args.destination.lower() in uid.lower()]
            destination = transceivers[nodes_suggestion[0]] \
                if len(nodes_suggestion)>0 else list(transceivers.values())[0]
            print(f'invalid destination node specified, did you mean:\
                \n{nodes_suggestion}?\
                \n{args.destination!r}, replaced with {destination.uid}')
    else:
        logger.info('No source node specified: picking random transceiver')
        destination = list(transceivers.values())[1]

    logger.info(f'source = {args.source!r}')
    logger.info(f'destination = {args.destination!r}')

    params = {}
    params['request_id'] = 0
    params['trx_type'] = ''
    params['trx_mode'] = ''
    params['source'] = source.uid
    params['destination'] = destination.uid
    params['nodes_list'] = [destination.uid]
    params['loose_list'] = ['strict']
    params['format'] = ''
    trx_params = trx_mode_params(equipment)
    if args.power:
        trx_params['power'] = db2lin(float(args.power))*1e-3
    params.update(trx_params)
    req = Path_request(**params)
    path = main(network, equipment, source, destination, req)

    if args.plot:
        plot_results(network, path, source, destination)
