#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
transmission_main_example.py
============================

Main example for transmission simulation.

Reads from network JSON (by default, `edfa_example_network.json`)
'''

from gnpy.core.equipment import load_equipment, trx_mode_params
from gnpy.core.utils import db2lin, lin2db, write_csv
from argparse import ArgumentParser
from sys import exit
from pathlib import Path
from json import loads
from collections import Counter
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
from numpy import linspace, mean
from matplotlib.pyplot import show, axis, figure, title, text
from networkx import (draw_networkx_nodes, draw_networkx_edges,
                      draw_networkx_labels, dijkstra_path)
from gnpy.core.network import load_network, build_network, save_network
from gnpy.core.elements import Transceiver, Fiber, Edfa, Roadm
from gnpy.core.info import create_input_spectral_information, SpectralInformation, Channel, Power, Pref
from gnpy.core.request import Path_request, RequestParams, compute_constrained_path, propagate2

logger = getLogger(__name__)

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

def plot_results(network, path, source, destination, infos):
    path_edges = set(zip(path[:-1], path[1:]))
    edges = set(network.edges()) - path_edges
    pos = {n: (n.lng, n.lat) for n in network.nodes()}
    nodes = {}
    for k, (x, y) in pos.items():
        nodes.setdefault((round(x, 0), round(y, 0)), []).append(k)
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

    textbox = text(0.85, 0.50, 'Spectral Information\n', fontsize=12, fontname='Ubuntu Mono',
                   verticalalignment='top', transform=fig.axes[0].transAxes,
                   bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    def hover(event):
        if event.xdata is None or event.ydata is None:
            return
        if fig.contains(event):
            x, y = round(event.xdata, 0), round(event.ydata, 0)
            if (x, y) in nodes:
                disp = [n for n in nodes[x, y] if n in infos]
                msg = "\n\n".join(str(n) for n in disp)
                msg = f'Spectral Information\n\n{msg}'
                textbox.set_text(msg)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)
    show()


def main(network, equipment, source, destination, req = None):
    result_dicts = {}
    network_data = [{
                    'network_name'  : str(args.filename),
                    'source'        : source.uid,
                    'destination'   : destination.uid
                    }]
    result_dicts.update({'network': network_data})
    design_data = [{
                    'power_mode'        : equipment['Spans']['default'].power_mode,
                    'span_power_range'  : equipment['Spans']['default'].delta_power_range_db,
                    'design_pch'        : equipment['SI']['default'].power_dbm,
                    'baud_rate'         : equipment['SI']['default'].baud_rate
                    }]
    result_dicts.update({'design': design_data})
    simulation_data = []
    result_dicts.update({'simulation results': simulation_data})

    power_mode = equipment['Spans']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     f'=> it can be modified in eqpt_config.json - Spans']))

    pref_ch_db = lin2db(req.power*1e3) #reference channel power / span (SL=20dB)
    pref_total_db = pref_ch_db + lin2db(req.nb_channel) #reference total power / span (SL=20dB)
    build_network(network, equipment, pref_ch_db, pref_total_db)
    path = compute_constrained_path(network, req)

    spans = [s.length for s in path if isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans):.0f}m between {source.uid} and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')

    try:
        p_start, p_stop, p_step = equipment['SI']['default'].power_range_db
        p_num = abs(int(round((p_stop - p_start)/p_step))) + 1 if p_step != 0 else 1
        power_range = list(linspace(p_start, p_stop, p_num))
    except TypeError:
        print('invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')
        power_range = [0]

    for dp_db in power_range:
        req.power = db2lin(pref_ch_db + dp_db)*1e-3
        print(f'\nPropagating with input power = {lin2db(req.power*1e3):.2f}dBm :')
        infos = propagate2(path, req, equipment, show=len(power_range)==1)
        print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f}dBm :')
        print(destination)

        #print(f'\n !!!!!!!!!!!!!!!!!     TEST POINT         !!!!!!!!!!!!!!!!!!!!!')
        #print(f'carriers ase output of {path[1]} =\n {list(path[1].carriers("out", "nli"))}')
        # => use "in" or "out" parameter
        # => use "nli" or "ase" or "signal" or "total" parameter

        simulation_data.append({
                    'Pch_dBm'               : pref_ch_db + dp_db,
                    'OSNR_ASE_0.1nm'        : round(mean(destination.osnr_ase_01nm),2),
                    'OSNR_ASE_signal_bw'    : round(mean(destination.osnr_ase),2),
                    'SNR_nli_signal_bw'     : round(mean(destination.osnr_nli),2),
                    'SNR_total_signal_bw'   : round(mean(destination.snr),2)
                            })
    write_csv(result_dicts, 'simulation_result.csv')
    return path, infos


parser = ArgumentParser()
parser.add_argument('-e', '--equipment', type=Path,
                    default=Path(__file__).parent / 'eqpt_config.json')
parser.add_argument('-pl', '--plot', action='store_true')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increases verbosity for each occurence')
parser.add_argument('-l', '--list-nodes', action='store_true', help='list all transceiver nodes')
parser.add_argument('-po', '--power', default=0, help='channel ref power in dBm')
parser.add_argument('-names', '--names-matching', action='store_true', help='display network names that are closed matches')
#parser.add_argument('-plb', '--power-lower-bound', default=0, help='power sweep lower bound')
#parser.add_argument('-pub', '--power-upper-bound', default=1, help='power sweep upper bound')
parser.add_argument('filename', nargs='?', type=Path,
                    default=Path(__file__).parent / 'edfa_example_network.json')
parser.add_argument('source', nargs='?', help='source node')
parser.add_argument('destination',   nargs='?', help='destination node')


if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={0: ERROR, 1: INFO, 2: DEBUG}.get(args.verbose, DEBUG))

    equipment = load_equipment(args.equipment)
    network = load_network(args.filename, equipment, args.names_matching)

    if args.plot:
        plot_baseline(network)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}

    if not transceivers:
        exit('Network has no transceivers!')
    if len(transceivers) < 2:
        exit('Network has only one transceiver!')

    if args.list_nodes:
        for uid in transceivers:
            print(uid)
        exit()

    #First try to find exact match if source/destination provided
    if args.source:
        source = transceivers.pop(args.source, None)
        valid_source = True if source else False
    else:
        source = None
        logger.info('No source node specified: picking random transceiver')

    if args.destination:
        destination = transceivers.pop(args.destination, None)
        valid_destination = True if destination else False
    else:
        destination = None
        logger.info('No destination node specified: picking random transceiver')

    #If no exact match try to find partial match
    if args.source and not source:
        #TODO code a more advanced regex to find nodes match
        source = next((transceivers.pop(uid) for uid in transceivers \
                  if args.source.lower() in uid.lower()), None)

    if args.destination and not destination:
        #TODO code a more advanced regex to find nodes match
        destination = next((transceivers.pop(uid) for uid in transceivers \
                  if args.destination.lower() in uid.lower()), None)

    #If no partial match or no source/destination provided pick random
    if not source:
        source = list(transceivers.values())[0]
        del transceivers[source.uid]

    if not destination:
        destination = list(transceivers.values())[0]

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
    params['path_bandwidth'] = 0
    trx_params = trx_mode_params(equipment)
    if args.power:
        trx_params['power'] = db2lin(float(args.power))*1e-3
    params.update(trx_params)
    req = Path_request(**params)
    path, infos = main(network, equipment, source, destination, req)
    save_network(args.filename, network)

    if not args.source:
        print(f'\n(No source node specified: picked {source.uid})')
    elif not valid_source:
        print(f'\n(Invalid source node {args.source!r} replaced with {source.uid})')

    if not args.destination:
        print(f'\n(No destination node specified: picked {destination.uid})')
    elif not valid_destination:
        print(f'\n(Invalid destination node {args.destination!r} replaced with {destination.uid})')

    if args.plot:
        plot_results(network, path, source, destination, infos)
