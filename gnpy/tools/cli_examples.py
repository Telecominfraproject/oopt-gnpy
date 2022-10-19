#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.tools.cli_examples
=======================

Common code for CLI examples
"""

import argparse
import logging
import sys
from math import ceil
from numpy import linspace, mean
from pathlib import Path

import gnpy.core.ansi_escapes as ansi_escapes
from gnpy.core.elements import Transceiver, Fiber, RamanFiber
from gnpy.core.equipment import trx_mode_params
import gnpy.core.exceptions as exceptions
from gnpy.core.network import add_missing_elements_in_network, design_network
from gnpy.core.parameters import SimParams
from gnpy.core.utils import db2lin, lin2db, automatic_nch, watt2dbm, dbm2watt
from gnpy.topology.request import (ResultElement, jsontocsv, compute_path_dsjctn, requests_aggregation,
                                   BLOCKING_NOPATH, correct_json_route_list,
                                   deduplicate_disjunctions, compute_path_with_disjunction,
                                   PathRequest, compute_constrained_path, propagate)
from gnpy.topology.spectrum_assignment import build_oms_list, pth_assign_spectrum
from gnpy.tools.json_io import (load_equipment, load_network, load_json, load_requests, save_network,
                                requests_from_json, disjunctions_from_json, save_json, load_initial_spectrum)
from gnpy.tools.plots import plot_baseline, plot_results

_logger = logging.getLogger(__name__)
_examples_dir = Path(__file__).parent.parent / 'example-data'
_help_footer = '''
This program is part of GNPy, https://github.com/TelecomInfraProject/oopt-gnpy

Learn more at https://gnpy.readthedocs.io/

'''
_help_fname_json = 'FILE.json'
_help_fname_json_csv = 'FILE.(json|csv)'


def show_example_data_dir():
    print(f'{_examples_dir}/')


def load_common_data(equipment_filename, topology_filename, simulation_filename, save_raw_network_filename):
    """Load common configuration from JSON files"""

    try:
        equipment = load_equipment(equipment_filename)
        network = load_network(topology_filename, equipment)
        if save_raw_network_filename is not None:
            save_network(network, save_raw_network_filename)
            print(f'{ansi_escapes.blue}Raw network (no optimizations) saved to {save_raw_network_filename}{ansi_escapes.reset}')
        if not simulation_filename:
            sim_params = {}
            if next((node for node in network if isinstance(node, RamanFiber)), None) is not None:
                print(f'{ansi_escapes.red}Invocation error:{ansi_escapes.reset} '
                      f'RamanFiber requires passing simulation params via --sim-params')
                sys.exit(1)
        else:
            sim_params = load_json(simulation_filename)
        SimParams.set_params(sim_params)
    except exceptions.EquipmentConfigError as e:
        print(f'{ansi_escapes.red}Configuration error in the equipment library:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ParametersError as e:
        print(f'{ansi_escapes.red}Simulation parameters error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
        sys.exit(1)

    return (equipment, network)


def _setup_logging(args):
    logging.basicConfig(level={2: logging.DEBUG, 1: logging.INFO, 0: logging.WARNING}.get(args.verbose, logging.DEBUG))


def _add_common_options(parser: argparse.ArgumentParser, network_default: Path):
    parser.add_argument('topology', nargs='?', type=Path, metavar='NETWORK-TOPOLOGY.(json|xls|xlsx)',
                        default=network_default,
                        help='Input network topology')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be specified several times)')
    parser.add_argument('-e', '--equipment', type=Path, metavar=_help_fname_json,
                        default=_examples_dir / 'eqpt_config.json', help='Equipment library')
    parser.add_argument('--sim-params', type=Path, metavar=_help_fname_json,
                        default=None, help='Path to the JSON containing simulation parameters (required for Raman). '
                                           f'Example: {_examples_dir / "sim_params.json"}')
    parser.add_argument('--save-network', type=Path, metavar=_help_fname_json,
                        help='Save the final network as a JSON file')
    parser.add_argument('--save-network-before-autodesign', type=Path, metavar=_help_fname_json,
                        help='Dump the network into a JSON file prior to autodesign')
    parser.add_argument('--no-insert-edfas', action='store_true',
                        help='Disable insertion of EDFAs after ROADMs and fibers '
                             'as well as splitting of fibers by auto-design.')


def transmission_main_example(args=None):
    parser = argparse.ArgumentParser(
        description='Send a full spectrum load through the network from point A to point B',
        epilog=_help_footer,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    _add_common_options(parser, network_default=_examples_dir / 'edfa_example_network.json')
    parser.add_argument('--show-channels', action='store_true', help='Show final per-channel OSNR and GSNR summary')
    parser.add_argument('-pl', '--plot', action='store_true')
    parser.add_argument('-l', '--list-nodes', action='store_true', help='list all transceiver nodes')
    parser.add_argument('-po', '--power', default=0, help='channel ref power in dBm')
    parser.add_argument('--spectrum', type=Path, help='user defined mixed rate spectrum JSON file')
    parser.add_argument('source', nargs='?', help='source node')
    parser.add_argument('destination', nargs='?', help='destination node')

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

    (equipment, network) = load_common_data(args.equipment, args.topology, args.sim_params, args.save_network_before_autodesign)

    if args.plot:
        plot_baseline(network)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, Transceiver)}

    if not transceivers:
        sys.exit('Network has no transceivers!')
    if len(transceivers) < 2:
        sys.exit('Network has only one transceiver!')

    if args.list_nodes:
        for uid in transceivers:
            print(uid)
        sys.exit()

    # First try to find exact match if source/destination provided
    if args.source:
        source = transceivers.pop(args.source, None)
        valid_source = True if source else False
    else:
        source = None
        _logger.info('No source node specified: picking random transceiver')

    if args.destination:
        destination = transceivers.pop(args.destination, None)
        valid_destination = True if destination else False
    else:
        destination = None
        _logger.info('No destination node specified: picking random transceiver')

    # If no exact match try to find partial match
    if args.source and not source:
        # TODO code a more advanced regex to find nodes match
        source = next((transceivers.pop(uid) for uid in transceivers
                       if args.source.lower() in uid.lower()), None)

    if args.destination and not destination:
        # TODO code a more advanced regex to find nodes match
        destination = next((transceivers.pop(uid) for uid in transceivers
                            if args.destination.lower() in uid.lower()), None)

    # If no partial match or no source/destination provided pick random
    if not source:
        source = list(transceivers.values())[0]
        del transceivers[source.uid]

    if not destination:
        destination = list(transceivers.values())[0]

    _logger.info(f'source = {args.source!r}')
    _logger.info(f'destination = {args.destination!r}')

    params = {}
    params['request_id'] = 0
    params['trx_type'] = ''
    params['trx_mode'] = ''
    params['source'] = source.uid
    params['destination'] = destination.uid
    params['bidir'] = False
    params['nodes_list'] = [destination.uid]
    params['loose_list'] = ['strict']
    params['format'] = ''
    params['path_bandwidth'] = 0
    params['effective_freq_slot'] = None
    trx_params = trx_mode_params(equipment)
    if args.power:
        trx_params['power'] = db2lin(float(args.power)) * 1e-3
    params.update(trx_params)
    initial_spectrum = None
    params['nb_channel'] = automatic_nch(trx_params['f_min'], trx_params['f_max'], trx_params['spacing'])
    # use ref_req to hold reference channel used for design and req for the propagation
    # and req to hold channels to be propagated
    # apply power sweep on the design and on the channels
    ref_req = PathRequest(**params)
    pref_ch_db = watt2dbm(ref_req.power)
    if args.spectrum:
        # use the spectrum defined by user for the propagation.
        # the nb of channel for design remains the one of the reference channel
        initial_spectrum = load_initial_spectrum(args.spectrum)
        params['nb_channel'] = len(initial_spectrum)
        print('User input for spectrum used for propagation instead of SI')
    req = PathRequest(**params)
    p_ch_db = watt2dbm(req.power)
    req.initial_spectrum = initial_spectrum
    print(f'There are {req.nb_channel} channels propagating')
    power_mode = equipment['Span']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     '=> it can be modified in eqpt_config.json - Span']))
    if not args.no_insert_edfas:
        try:
            add_missing_elements_in_network(network, equipment)
        except exceptions.NetworkTopologyError as e:
            print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
            sys.exit(1)
        except exceptions.ConfigurationError as e:
            print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
            sys.exit(1)

    path = compute_constrained_path(network, req)
    spans = [s.params.length for s in path if isinstance(s, RamanFiber) or isinstance(s, Fiber)]
    power_range = [0]
    if power_mode:
        # power cannot be changed in gain mode
        try:
            p_start, p_stop, p_step = equipment['SI']['default'].power_range_db
            p_num = abs(int(round((p_stop - p_start) / p_step))) + 1 if p_step != 0 else 1
            power_range = list(linspace(p_start, p_stop, p_num))
        except TypeError:
            print('invalid power range definition in eqpt_config, should be power_range_db: [lower, upper, step]')
    # initial network is designed using req.power. that is that any missing information (amp gain or delta_p) is filled
    # using this req.power, previous to any sweep requested later on.
    try:
        design_network(ref_req, network, equipment, set_connector_losses=True, verbose=True)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)

    print(f'\nThere are {len(spans)} fiber spans over {sum(spans)/1000:.0f} km between {source.uid} '
          f'and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')
    for dp_db in power_range:
        ref_req.power = dbm2watt(pref_ch_db + dp_db)
        req.power = dbm2watt(p_ch_db + dp_db)
        design_network(ref_req, network, equipment, set_connector_losses=False, verbose=False)
        # if initial spectrum did not contain any power, now we need to use this one.
        # note the initial power defines a differential wrt req.power so that if req.power is set to 2mW (3dBm)
        # and initial spectrum was set to 0, this sets a initial per channel delta power to -3dB, so that
        # whatever the equalization, -3 dB is applied on all channels (ie initial power in initial spectrum pre-empts
        # "--power" option)
        if power_mode:
            print(f'\nPropagating with input power = {ansi_escapes.cyan}{watt2dbm(req.power):.2f} '
                  + f'dBm{ansi_escapes.reset}:')
        else:
            print(f'\nPropagating in {ansi_escapes.cyan}gain mode{ansi_escapes.reset}: power cannot be set manually')
        infos = propagate(path, req, equipment)
        if len(power_range) == 1:
            for elem in path:
                print(elem)
            if power_mode:
                print(f'\nTransmission result for input power = {lin2db(req.power*1e3):.2f} dBm:')
            else:
                print(f'\nTransmission results:')
            print(f'  Final GSNR (0.1 nm): {ansi_escapes.cyan}{mean(destination.snr_01nm):.02f} dB{ansi_escapes.reset}')
        else:
            print(path[-1])

    if args.save_network is not None:
        save_network(network, args.save_network)
        print(f'{ansi_escapes.blue}Network (after autodesign) saved to {args.save_network}{ansi_escapes.reset}')

    if args.show_channels:
        print('\nThe GSNR per channel at the end of the line is:')
        print(
            '{:>5}{:>26}{:>26}{:>28}{:>28}{:>28}' .format(
                'Ch. #',
                'Channel frequency (THz)',
                'Channel power (dBm)',
                'OSNR ASE (signal bw, dB)',
                'SNR NLI (signal bw, dB)',
                'GSNR (signal bw, dB)'))
        for final_carrier, ch_osnr, ch_snr_nl, ch_snr in zip(
                infos.carriers, path[-1].osnr_ase, path[-1].osnr_nli, path[-1].snr):
            ch_freq = final_carrier.frequency * 1e-12
            ch_power = lin2db(final_carrier.power.signal * 1e3)
            print(
                '{:5}{:26.5f}{:26.2f}{:28.2f}{:28.2f}{:28.2f}' .format(
                    final_carrier.channel_number, round(
                        ch_freq, 5), round(
                        ch_power, 2), round(
                        ch_osnr, 2), round(
                        ch_snr_nl, 2), round(
                            ch_snr, 2)))

    if not args.source:
        print(f'\n(No source node specified: picked {source.uid})')
    elif not valid_source:
        print(f'\n(Invalid source node {args.source!r} replaced with {source.uid})')

    if not args.destination:
        print(f'\n(No destination node specified: picked {destination.uid})')
    elif not valid_destination:
        print(f'\n(Invalid destination node {args.destination!r} replaced with {destination.uid})')

    if args.plot:
        plot_results(network, path, source, destination)


def _path_result_json(pathresult):
    return {'response': [n.json for n in pathresult]}


def path_requests_run(args=None):
    parser = argparse.ArgumentParser(
        description='Compute performance for a list of services provided in a json file or an excel sheet',
        epilog=_help_footer,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    _add_common_options(parser, network_default=_examples_dir / 'meshTopologyExampleV2.xls')
    parser.add_argument('service_filename', nargs='?', type=Path, metavar='SERVICES-REQUESTS.(json|xls|xlsx)',
                        default=_examples_dir / 'meshTopologyExampleV2.xls',
                        help='Input service file')
    parser.add_argument('-bi', '--bidir', action='store_true',
                        help='considers that all demands are bidir')
    parser.add_argument('-o', '--output', type=Path, metavar=_help_fname_json_csv,
                        help='Store satisifed requests into a JSON or CSV file')

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

    _logger.info(f'Computing path requests {args.service_filename.name} into JSON format')

    (equipment, network) = load_common_data(args.equipment, args.topology, args.sim_params, args.save_network_before_autodesign)

    # Build the network once using the default power defined in SI in eqpt config
    # TODO power density: db2linp(ower_dbm": 0)/power_dbm": 0 * nb channels as defined by
    # spacing, f_min and f_max
    if not args.no_insert_edfas:
        try:
            add_missing_elements_in_network(network, equipment)
        except exceptions.NetworkTopologyError as e:
            print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
            sys.exit(1)
        except exceptions.ConfigurationError as e:
            print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
            sys.exit(1)

    params = {
        'request_id': 'reference',
        'trx_type': '',
        'trx_mode': '',
        'source': None,
        'destination': None,
        'bidir': False,
        'nodes_list': [],
        'loose_list': [],
        'format': '',
        'path_bandwidth': 0,
        'effective_freq_slot': None,
        'nb_channel': automatic_nch(equipment['SI']['default'].f_min, equipment['SI']['default'].f_max,
                                    equipment['SI']['default'].spacing)
    }
    trx_params = trx_mode_params(equipment)
    params.update(trx_params)
    reference_channel = PathRequest(**params)
    try:
        design_network(reference_channel, network, equipment, verbose=True)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)

    if args.save_network is not None:
        save_network(network, args.save_network)
        print(f'{ansi_escapes.blue}Network (after autodesign) saved to {args.save_network}{ansi_escapes.reset}')
    oms_list = build_oms_list(network, equipment)

    try:
        data = load_requests(args.service_filename, equipment, bidir=args.bidir,
                             network=network, network_filename=args.topology)
        rqs = requests_from_json(data, equipment)
    except exceptions.ServiceError as e:
        print(f'{ansi_escapes.red}Service error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    # check that request ids are unique. Non unique ids, may
    # mess the computation: better to stop the computation
    all_ids = [r.request_id for r in rqs]
    if len(all_ids) != len(set(all_ids)):
        for item in list(set(all_ids)):
            all_ids.remove(item)
        msg = f'Requests id {all_ids} are not unique'
        _logger.critical(msg)
        sys.exit()
    rqs = correct_json_route_list(network, rqs)

    # pths = compute_path(network, equipment, rqs)
    dsjn = disjunctions_from_json(data)

    print(f'{ansi_escapes.blue}List of disjunctions{ansi_escapes.reset}')
    print(dsjn)
    # need to warn or correct in case of wrong disjunction form
    # disjunction must not be repeated with same or different ids
    dsjn = deduplicate_disjunctions(dsjn)

    # Aggregate demands with same exact constraints
    print(f'{ansi_escapes.blue}Aggregating similar requests{ansi_escapes.reset}')

    rqs, dsjn = requests_aggregation(rqs, dsjn)
    # TODO export novel set of aggregated demands in a json file

    print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
    print(rqs)

    print(f'{ansi_escapes.blue}Computing all paths with constraints{ansi_escapes.reset}')
    try:
        pths = compute_path_dsjctn(network, equipment, rqs, dsjn)
    except exceptions.DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        sys.exit(1)

    print(f'{ansi_escapes.blue}Propagating on selected path{ansi_escapes.reset}')
    propagatedpths, reversed_pths, reversed_propagatedpths = compute_path_with_disjunction(network, equipment, rqs, pths)
    # Note that deepcopy used in compute_path_with_disjunction returns
    # a list of nodes which are not belonging to network (they are copies of the node objects).
    # so there can not be propagation on these nodes.

    pth_assign_spectrum(pths, rqs, oms_list, reversed_pths)

    print(f'{ansi_escapes.blue}Result summary{ansi_escapes.reset}')
    header = ['req id', '  demand', ' GSNR@bandwidth A-Z (Z-A)', ' GSNR@0.1nm A-Z (Z-A)',
              '  Receiver minOSNR', '  mode', '  Gbit/s', '  nb of tsp pairs',
              'N,M or blocking reason']
    data = []
    data.append(header)
    for i, this_p in enumerate(propagatedpths):
        rev_pth = reversed_propagatedpths[i]
        if rev_pth and this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)} ({round(mean(rev_pth[-1].snr),2)})'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' +\
                f' ({round(mean(rev_pth[-1].snr_01nm),2)})'
        elif this_p:
            psnrb = f'{round(mean(this_p[-1].snr),2)}'
            psnr = f'{round(mean(this_p[-1].snr_01nm),2)}'

        try:
            if rqs[i].blocking_reason in BLOCKING_NOPATH:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} :',
                        f'-', f'-', f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',
                        f'-', f'{rqs[i].blocking_reason}']
            else:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                        psnr, f'-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        f'-', f'{rqs[i].blocking_reason}']
        except AttributeError:
            line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                    psnr, f'{rqs[i].OSNR + equipment["SI"]["default"].sys_margins}',
                    f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9,2)}',
                    f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate) }', f'({rqs[i].N},{rqs[i].M})']
        data.append(line)

    col_width = max(len(word) for row in data for word in row[2:])   # padding
    firstcol_width = max(len(row[0]) for row in data)   # padding
    secondcol_width = max(len(row[1]) for row in data)   # padding
    for row in data:
        firstcol = ''.join(row[0].ljust(firstcol_width))
        secondcol = ''.join(row[1].ljust(secondcol_width))
        remainingcols = ''.join(word.center(col_width, ' ') for word in row[2:])
        print(f'{firstcol} {secondcol} {remainingcols}')
    print(f'{ansi_escapes.yellow}Result summary shows mean GSNR and OSNR (average over all channels){ansi_escapes.reset}')

    if args.output:
        result = []
        # assumes that list of rqs and list of propgatedpths have same order
        for i, pth in enumerate(propagatedpths):
            result.append(ResultElement(rqs[i], pth, reversed_propagatedpths[i]))
        temp = _path_result_json(result)
        if args.output.suffix.lower() == '.json':
            save_json(temp, args.output)
            print(f'{ansi_escapes.blue}Saved JSON to {args.output}{ansi_escapes.reset}')
        elif args.output.suffix.lower() == '.csv':
            with open(args.output, "w", encoding='utf-8') as fcsv:
                jsontocsv(temp, equipment, fcsv)
            print(f'{ansi_escapes.blue}Saved CSV to {args.output}{ansi_escapes.reset}')
        else:
            print(f'{ansi_escapes.red}Cannot save output: neither JSON nor CSV file{ansi_escapes.reset}')
            sys.exit(1)
