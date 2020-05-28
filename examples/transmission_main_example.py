#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
transmission_main_example.py
============================

Main example for transmission simulation.

Reads from network JSON (by default, `edfa_example_network.json`)
'''

from argparse import ArgumentParser
from sys import exit
from pathlib import Path
from logging import getLogger, basicConfig, INFO, ERROR, DEBUG
from numpy import linspace, mean
from gnpy.core.equipment import trx_mode_params
from gnpy.core.network import build_network
from gnpy.core.elements import Transceiver, Fiber, RamanFiber
from gnpy.core.utils import db2lin, lin2db, write_csv
import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.topology.request import PathRequest, compute_constrained_path, propagate2
import gnpy.tools.cli_examples as cli_examples
from gnpy.tools.json_io import save_network
from gnpy.tools.plots import plot_baseline, plot_results


logger = getLogger(__name__)


def main(network, equipment, source, destination, req=None):
    result_dicts = {}
    network_data = [{
                    'network_name'  : str(args.filename),
                    'source'        : source.uid,
                    'destination'   : destination.uid
                    }]
    result_dicts.update({'network': network_data})
    design_data = [{
                    'power_mode'        : equipment['Span']['default'].power_mode,
                    'span_power_range'  : equipment['Span']['default'].delta_power_range_db,
                    'design_pch'        : equipment['SI']['default'].power_dbm,
                    'baud_rate'         : equipment['SI']['default'].baud_rate
                    }]
    result_dicts.update({'design': design_data})
    simulation_data = []
    result_dicts.update({'simulation results': simulation_data})

    power_mode = equipment['Span']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     f'=> it can be modified in eqpt_config.json - Span']))

    pref_ch_db = lin2db(req.power*1e3) #reference channel power / span (SL=20dB)
    pref_total_db = pref_ch_db + lin2db(req.nb_channel) #reference total power / span (SL=20dB)
    build_network(network, equipment, pref_ch_db, pref_total_db)
    path = compute_constrained_path(network, req)

    spans = [s.params.length for s in path if isinstance(s, RamanFiber) or isinstance(s, Fiber)]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans)/1000:.0f} km between {source.uid} '
          f'and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')

    try:
        p_start, p_stop, p_step = equipment['SI']['default'].power_range_db
        p_num = abs(int(round((p_stop - p_start)/p_step))) + 1 if p_step != 0 else 1
        power_range = list(linspace(p_start, p_stop, p_num))
    except TypeError:
        print('invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')
        power_range = [0]

    if not power_mode:
        #power cannot be changed in gain mode
        power_range = [0]
    for dp_db in power_range:
        req.power = db2lin(pref_ch_db + dp_db)*1e-3
        if power_mode:
            print(f'\nPropagating with input power = {ansi_escapes.cyan}{lin2db(req.power*1e3):.2f} dBm{ansi_escapes.reset}:')
        else:
            print(f'\nPropagating in {ansi_escapes.cyan}gain mode{ansi_escapes.reset}: power cannot be set manually')
        infos = propagate2(path, req, equipment)
        if len(power_range) == 1:
            for elem in path:
                print(elem)
            if power_mode:
                print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f} dBm:')
            else:
                print(f'\nTransmission results:')
            print(f'  Final SNR total (0.1 nm): {ansi_escapes.cyan}{mean(destination.snr_01nm):.02f} dB{ansi_escapes.reset}')
        else:
            print(path[-1])

        #print(f'\n !!!!!!!!!!!!!!!!!     TEST POINT         !!!!!!!!!!!!!!!!!!!!!')
        #print(f'carriers ase output of {path[1]} =\n {list(path[1].carriers("out", "nli"))}')
        # => use "in" or "out" parameter
        # => use "nli" or "ase" or "signal" or "total" parameter        
        if power_mode:
            simulation_data.append({
                        'Pch_dBm'               : pref_ch_db + dp_db,
                        'OSNR_ASE_0.1nm'        : round(mean(destination.osnr_ase_01nm),2),
                        'OSNR_ASE_signal_bw'    : round(mean(destination.osnr_ase),2),
                        'SNR_nli_signal_bw'     : round(mean(destination.osnr_nli),2),
                        'SNR_total_signal_bw'   : round(mean(destination.snr),2)
                                })
        else:
            simulation_data.append({
                        'gain_mode'             : 'power canot be set',
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
parser.add_argument('--sim-params', type=Path,
                    default=None, help='Path to the JSON containing simulation parameters (required for Raman)')
parser.add_argument('--show-channels', action='store_true', help='Show final per-channel OSNR summary')
parser.add_argument('-pl', '--plot', action='store_true')
parser.add_argument('-v', '--verbose', action='count', default=0, help='increases verbosity for each occurence')
parser.add_argument('-l', '--list-nodes', action='store_true', help='list all transceiver nodes')
parser.add_argument('-po', '--power', default=0, help='channel ref power in dBm')
parser.add_argument('-names', '--names-matching', action='store_true', help='display network names that are closed matches')
parser.add_argument('filename', nargs='?', type=Path,
                    default=Path(__file__).parent / 'edfa_example_network.json')
parser.add_argument('source', nargs='?', help='source node')
parser.add_argument('destination',   nargs='?', help='destination node')


if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={0: ERROR, 1: INFO, 2: DEBUG}.get(args.verbose, DEBUG))

    (equipment, network) = cli_examples.load_common_data(args.equipment, args.filename, args.sim_params,
                                                         fuzzy_name_matching=args.names_matching)

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
    params['bidir'] = False
    params['nodes_list'] = [destination.uid]
    params['loose_list'] = ['strict']
    params['exclude_nodes_list'] = []
    params['format'] = ''
    params['path_bandwidth'] = 0
    trx_params = trx_mode_params(equipment)
    if args.power:
        trx_params['power'] = db2lin(float(args.power))*1e-3
    params.update(trx_params)
    req = PathRequest(**params)
    path, infos = main(network, equipment, source, destination, req)
    save_network(args.filename, network)

    if args.show_channels:
        print('\nThe total SNR per channel at the end of the line is:')
        print('{:>5}{:>26}{:>26}{:>28}{:>28}{:>28}' \
            .format('Ch. #', 'Channel frequency (THz)', 'Channel power (dBm)', 'OSNR ASE (signal bw, dB)', 'SNR NLI (signal bw, dB)', 'SNR total (signal bw, dB)'))
        for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(infos[path[-1]][1].carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr):
            ch_freq = final_carrier.frequency * 1e-12
            ch_power = lin2db(final_carrier.power.signal*1e3)
            print('{:5}{:26.2f}{:26.2f}{:28.2f}{:28.2f}{:28.2f}' \
                .format(final_carrier.channel_number, round(ch_freq, 2), round(ch_power, 2), round(ch_osnr, 2), round(ch_snr_nl, 2), round(ch_snr, 2)))

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
