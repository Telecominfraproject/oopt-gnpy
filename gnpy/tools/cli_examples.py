#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.cli_examples: Common code for CLI examples
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.tools.cli_examples
=======================

Common code for CLI examples
"""

import argparse
import logging

from pathlib import Path
import sys
from typing import Union, List, Dict, Tuple
from math import ceil
from numpy import mean
from networkx import DiGraph
import pandas as pd
from tabulate import tabulate

from gnpy.core import ansi_escapes
# use an alias fro Transceiver import because autodoc from sphinx mixes json_io and elements Transceiver
from gnpy.core.elements import Transceiver as elementTransceiver, Fiber, RamanFiber, Roadm
from gnpy.core.equipment import trx_mode_params
from gnpy.core import exceptions
from gnpy.core.parameters import SimParams
from gnpy.core.utils import lin2db, pretty_summary_print, per_label_average, watt2dbm
from gnpy.topology.request import (ResultElement, jsontocsv, BLOCKING_NOPATH, PathRequest, correct_json_route_list)
from gnpy.tools.json_io import (load_equipments_and_configs, load_network, load_json, load_requests, save_network,
                                requests_from_json, save_json, load_initial_spectrum, DEFAULT_EQPT_CONFIG)
from gnpy.tools.plots import plot_baseline, plot_results
from gnpy.tools.worker_utils import designed_network, transmission_simulation, planning


_logger = logging.getLogger(__name__)
_examples_dir = Path(__file__).parent.parent / 'example-data'
_default_config_files = ['example-data/std_medium_gain_advanced_config.json',
                         'example-data/Juniper-BoosterHG.json',
                         'parameters.DEFAULT_EDFA_CONFIG']
_help_footer = '''
This program is part of GNPy, https://github.com/TelecomInfraProject/oopt-gnpy

Learn more at https://gnpy.readthedocs.io/

'''
_help_fname_json = 'FILE.json'
_help_fname_json_csv = 'FILE.(json|csv)'


def show_example_data_dir():
    """Print the example data directory path."""
    print(f'{_examples_dir}/')


def load_common_data(equipment_filename: Path,
                     extra_equipment_filenames: List[Path], extra_config_filenames: List[Path],
                     topology_filename: Path, simulation_filename: Path, save_raw_network_filename: Path):
    """Load common configuration from JSON files, merging additional equipment if provided.

    :param equipment_filename: Path to the main equipment configuration file.
    :type equipment_filename: Path
    :param extra_equipment_filenames: List of additional equipment configuration files.
    :type extra_equipment_filenames: List[Path]
    :param extra_config_filenames: List of additional configuration files.
    :type extra_config_filenames: List[Path]
    :param topology_filename: Path to the network topology file.
    :type topology_filename: Path
    :param simulation_filename: Path to the simulation parameters file.
    :type simulation_filename: Path
    :param save_raw_network_filename: Path to save the raw network configuration.
    :type save_raw_network_filename: Path
    :raises exceptions.EquipmentConfigError: If there is a configuration error in the equipment library.
    :raises exceptions.NetworkTopologyError: If the network definition is invalid.
    :raises exceptions.ParametersError: If there is an error with simulation parameters.
    :raises exceptions.ConfigurationError: If there is a general configuration error.
    :raises exceptions.ServiceError: If there is a service-related error.
    """
    try:
        equipment = load_equipments_and_configs(equipment_filename, extra_equipment_filenames, extra_config_filenames)
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


def _setup_logging(args: argparse.Namespace):
    """Set up logging based on verbosity level.

    :param args: The parsed command-line arguments.
    :type args: argparse.Namespace
    """
    logging.basicConfig(level={2: logging.DEBUG, 1: logging.INFO, 0: logging.WARNING}.get(args.verbose, logging.DEBUG))


def _add_common_options(parser: argparse.ArgumentParser, network_default: Path):
    """Add common command-line options to the argument parser.

    :param parser: The argument parser to which options will be added.
    :type parser: argparse.ArgumentParser
    :param network_default: The default path for the network topology file.
    :type network_default: Path
    """
    parser.add_argument('topology', nargs='?', type=Path, metavar='NETWORK-TOPOLOGY.(json|xls|xlsx)',
                        default=network_default,
                        help='Input network topology')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be specified several times)')
    parser.add_argument('-e', '--equipment', type=Path, metavar=_help_fname_json,
                        default=DEFAULT_EQPT_CONFIG, help='Equipment library')
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
    # Option for additional equipment files
    parser.add_argument('--extra-equipment', nargs='+', type=Path,
                        metavar=_help_fname_json, default=None,
                        help='List of additional equipment files to complement the main equipment file.')
    # Option for additional config files
    parser.add_argument('--extra-config', nargs='+', type=Path,
                        metavar=_help_fname_json,
                        help='List of additional config files as referenced in equipment files with '
                             '"advanced_config_from_json" or "default_config_from_json".'
                             f'Existing configs:\n{_default_config_files}')


def _infer_trx(target_roadm_uid: str, all_roadms: Dict, network: DiGraph):
    """Return the 'trx' of the 'target_roadm' (uid) among 'all_roadm's (dict) of 'network'.

    :param target_roadm_uid: The UID of the target ROADM.
    :type target_roadm_uid: str
    :param all_roadms: A dictionary of all ROADMs in the network.
    :type all_roadms: Dict
    :param network: The network object.
    :type network: DiGraph
    :return: The Transceiver object associated with the target ROADM.
    :raises exceptions.NetworkTopologyError: If no transceiver can be associated with the target ROADM.
    """
    target_trx = [n for n in network.successors(all_roadms[target_roadm_uid]) if isinstance(n, elementTransceiver)]
    if not target_trx:
        msg = f'Could not associate a transceiver with node: {target_roadm_uid}'
        raise exceptions.NetworkTopologyError(msg)
    target = target_trx[0]
    msg = f'Picking transceiver from provided path: {target.uid}'
    print(msg)
    _logger.info(msg)
    return target


def _get_nodes_from_path(path_list: List[str], all_roadms: Dict) -> List[str]:
    """
    Return a Nodes-list made from the parsed 'path_string'
    and which are part of 'all_roadms'.

    :param path_list: A list of path elements to check.
    :type path_list: List[str]
    :param all_roadms: A dictionary of all ROADMs in the network.
    :type all_roadms: dict
    :return: A list of nodes corresponding to the path elements.
    :rtype: List[str]
    :raises exceptions.NetworkTopologyError: If some elements in the requested path could not be resolved.
    """
    path_check = set(path_list)
    # check if all elements of the path are valid
    for p in set(path_list):
        for r in all_roadms.keys():
            if p.lower() in r.lower():
                path_check.remove(p)

    if not path_check:
        # build the requested path using the identified Roadms
        nodes = [r for p in path_list for r in all_roadms.keys() if p.lower() in r.lower()]
        return nodes
    msg = f'Some element(s) in requested path could not be resolved: {path_check}'
    raise exceptions.NetworkTopologyError(msg)


def _get_params_from_path(path_raw: List[str], network, source: elementTransceiver, destination: elementTransceiver,
                          args_source: str, args_destination: str) -> Tuple[elementTransceiver, elementTransceiver,
                                                                            List[str], List[str]]:
    """Extract the explicit path from the provided raw path argument.

    :param path_raw: A list of raw path elements.
    :type path_raw: List[str]
    :param network: The network object.
    :type network: Any
    :param source: The source Transceiver object.
    :type source: elementTransceiver
    :param destination: The destination elementTransceiver object.
    :type destination: elementTransceiver
    :param args_source: The command-line argument for the source node.
    :type args_source: str
    :param args_destination: The command-line argument for the destination node.
    :type args_destination: str
    :return: A tuple containing the source, destination, nodes list, and loose list.
    :rtype: Tuple[elementTransceiver, elementTransceiver, List[str], List[str]]
    :raises exceptions.NetworkTopologyError: If some elements in the requested path could not be resolved.
    """
    # List all roadm nodes that can possible be part of the path
    all_roadms = {n.uid: n for n in network.nodes() if isinstance(n, Roadm)}
    # Verify that each name in path exists in this list. The check is case insensitive and with loose match
    # to ease user experience
    path_neg = set(path_raw)
    for p in set(path_raw):
        for r in all_roadms.keys():
            if p.lower() in r.lower():
                path_neg.remove(p)
    if path_neg:
        msg = f'Some element(s) in requested path could not be resolved: {path_neg}'
        raise exceptions.NetworkTopologyError(msg)
    nodes = _get_nodes_from_path(path_raw, all_roadms)
    # inferring missing source transceiver from first roadm in path
    if not args_source:
        source_roadm_uid = nodes[0].strip()
        source = _infer_trx(source_roadm_uid, all_roadms, network)

    # inferring missing destination transceiver from last roadm in path
    if not args_destination:
        destination_roadm_uid = nodes[-1].strip()
        destination = _infer_trx(destination_roadm_uid, all_roadms, network)
    nodes_list = nodes + [destination.uid]
    loose_list = len(nodes_list) * ['STRICT']
    return source, destination, nodes_list, loose_list


def _get_rq_from_service(service: Path, route_id: str, network, equipment,
                         args_topology: Path) -> Tuple[elementTransceiver, elementTransceiver, PathRequest]:
    """Retrieve the request from the service file.

    :param service: The path to the service request file.
    :type service: Path
    :param route_id: The ID of the route to retrieve.
    :type route_id: str
    :param network: The network object.
    :type network: Any
    :param equipment: The equipment configuration.
    :type equipment: Any
    :param args_topology: The path to the topology file.
    :type args_topology: Path
    :return: A tuple containing the source and destination Transceiver objects, and the request.
    :rtype: Tuple[elementTransceiver, elementTransceiver, PathRequest]
    :raises exceptions.ServiceError: If the requested route_id could not be found.
    """
    data = load_requests(service, equipment, bidir=False, network=network, network_filename=args_topology)
    rqs = requests_from_json(data, equipment)
    all_ids = [r.request_id for r in rqs]

    if route_id not in all_ids:
        msg = f'Requested route_id \'{route_id}\' could not be found among {all_ids} of the Service-sheet \'{service}\''
        raise exceptions.ServiceError(msg)

    # picking the request matching the selected route_id
    rqs = [rqs[all_ids.index(route_id)]]
    # correct as done in path_request_run()
    rqs = correct_json_route_list(network, rqs)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, elementTransceiver)}
    # find the proper trx of source and destination
    source = next((transceivers.pop(uid) for uid in transceivers
                   if (rqs[0].source).lower() in uid.lower()), None)
    destination = next((transceivers.pop(uid) for uid in transceivers
                        if (rqs[0].destination).lower() in uid.lower()), None)

    # ensure a 'Mode' is defined for the Service
    if not rqs[0].tsp_mode:
        msg = f'Missing "Mode" for Service "{rqs[0].request_id}" in the Service-file {service}'
        _logger.warning(msg)
        modes = equipment['Transceiver'][rqs[0].tsp].mode

        # Sort compatible modes by decreasing baud_rate
        modes_sorted = sorted({(this_mode['baud_rate'], this_mode['equalization_offset_db'])
                               for this_mode in modes
                               if float(this_mode['min_spacing']) <= rqs[0].spacing}, reverse=True)
        if modes_sorted:
            this_br, _ = modes_sorted[0]
            modes_to_explore = [this_mode for this_mode in modes
                                if abs(this_mode['baud_rate'] - this_br) <= 5.0e9
                                and float(this_mode['min_spacing']) <= rqs[0].spacing]
            modes_to_explore = sorted(modes_to_explore,
                                      key=lambda x: (x['bit_rate'], x['equalization_offset_db']), reverse=True)
            trx_params = trx_mode_params(equipment, rqs[0].tsp, modes_to_explore[0]['format'], True)
            rqs[0].tsp_mode = trx_params['format']
            rqs[0].format = trx_params['format']
            rqs[0].baud_rate = trx_params['baud_rate']
            rqs[0].roll_off = trx_params['roll_off']
            rqs[0].bit_rate = trx_params['bit_rate']
            rqs[0].OSNR = trx_params['OSNR']
            rqs[0].tx_osnr = trx_params['tx_osnr']
            rqs[0].min_spacing = trx_params['min_spacing']
            rqs[0].cost = trx_params['cost']
            rqs[0].penalties = trx_params['penalties']
            rqs[0].cost = trx_params['cost']
            rqs[0].offset_db = trx_params['equalization_offset_db']
            mode_name = trx_params['format']
            baud_rate_ghz = rqs[0].baud_rate * 1e-9  # Convert to GHz
            spacing_ghz = rqs[0].spacing * 1e-9  # Convert to GHz
            msg = 'Choosing first compatible mode.' \
                + f'Selected mode: {mode_name}, Baud rate: {baud_rate_ghz:.2f} GHz, Spacing: {spacing_ghz:.2f} GHz'
            _logger.warning(msg)

        else:
            raise exceptions.ServiceError("No compatible mode found.")

    return source, destination, rqs[0]


def transmission_main_example(args: Union[List[str], None] = None):
    """Main script running a single simulation. It returns the detailed power across crossed elements and
    average performance accross all channels.

    :param args: Command-line arguments (default is None).
    :type args: Union[List[str], None]
    """
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
    parser.add_argument('-p', '--path', type=str, nargs='?', help='Compute for the given path succession of ROADM '
                        + 'element name(s) where the separating element can be "|" or ","')
    parser.add_argument('-s', '--service', nargs='?', type=Path, required='--route_id' in sys.argv or '-r' in sys.argv,
                        metavar='SERVICES-REQUESTS.(json|xls|xlsx)', help='Input Service-file')
    parser.add_argument('-r', '--route_id', nargs='?', required='--service' in sys.argv or '-s' in sys.argv,
                        help='Compute for the given route-id of the Service')

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

    (equipment, network) = load_common_data(args.equipment, args.extra_equipment, args.extra_config, args.topology,
                                            args.sim_params, args.save_network_before_autodesign)

    if args.plot:
        plot_baseline(network)

    transceivers = {n.uid: n for n in network.nodes() if isinstance(n, elementTransceiver)}

    if not transceivers:
        sys.exit('Network has no transceivers!')
    if len(transceivers) < 2:
        sys.exit('Network has only one transceiver!')

    if args.list_nodes:
        for uid in transceivers:
            print(uid)
        sys.exit()

    # First try to find exact match if source/destination provided
    source = None
    valid_source = None
    if args.source:
        source = transceivers.pop(args.source, None)
        valid_source = bool(source)

    destination = None
    valid_destination = None
    nodes_list = []
    loose_list = []
    if args.destination:
        destination = transceivers.pop(args.destination, None)
        valid_destination = bool(destination)

    # If no exact match try to find partial match
    if args.source and not source:
        # Do not code a more advanced regex to find nodes match (keep simple behaviour)
        source = next((transceivers.pop(uid) for uid in transceivers
                       if args.source.lower() in uid.lower()), None)

    if args.destination and not destination:
        # Do not code a more advanced regex to find nodes match (keep simple behaviour)
        destination = next((transceivers.pop(uid) for uid in transceivers
                            if args.destination.lower() in uid.lower()), None)
        nodes_list = [destination.uid]
        loose_list = ['STRICT']

    if args.path:
        print(f'Requested path: {args.path}')
        # the path elements can be separated by '|' or ','
        if '|' in args.path:
            path_raw = [k.strip() for k in (args.path).split('|')]
        elif ',' in args.path:
            path_raw = [k.strip() for k in (args.path).split(',')]
        else:
            path_raw = [(args.path).strip()]

        source, destination, nodes_list, loose_list = \
            _get_params_from_path(path_raw, network, source, destination, args.source, args.destination)

    if args.route_id and not args.service:
        raise exceptions.ServiceError(f'Requested route_id {args.route_id} requires a Service-file')

    rq = None
    if args.service:
        print(f'Requested route_id: {args.route_id}')
        service = args.service
        try:
            source, destination, rq = _get_rq_from_service(service, args.route_id, network, equipment, args.topology)
            nodes_list = rq.nodes_list.append(destination.uid)
            loose_list = rq.loose_list.append('STRICT')
        except exceptions.ServiceError as e:
            raise exceptions.ServiceError(f'Service error: {e}')

    # If no partial match or no source/destination provided pick random
    if not source:
        source = list(transceivers.values())[0]
        del transceivers[source.uid]
        _logger.info('No source node specified: picking random transceiver')

    if not destination:
        destination = list(transceivers.values())[0]
        nodes_list = [destination.uid]
        loose_list = ['STRICT']
        _logger.info('No destination node specified: picking random transceiver')

    _logger.info(f'source = {source.uid!r}')
    _logger.info(f'destination = {destination.uid!r}')

    initial_spectrum = None
    if args.spectrum:
        # use the spectrum defined by user for the propagation.
        # the nb of channel for design remains the one of the reference channel
        initial_spectrum = load_initial_spectrum(args.spectrum)
        print('User input for spectrum used for propagation instead of SI')
    power_mode = equipment['Span']['default'].power_mode
    print('\n'.join([f'Power mode is set to {power_mode}',
                     '=> it can be modified in eqpt_config.json - Span']))

    # Simulate !
    try:
        network, req, ref_req = designed_network(equipment, network, source.uid, destination.uid,
                                                 nodes_list=nodes_list, loose_list=loose_list,
                                                 args_power=args.power,
                                                 initial_spectrum=initial_spectrum,
                                                 no_insert_edfas=args.no_insert_edfas)
        path, propagations_for_path, powers_dbm, infos = transmission_simulation(equipment, network, req, ref_req)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ServiceError as e:
        print(f'Service error: {e}')
        sys.exit(1)
    except ValueError:
        sys.exit(1)
    # print or export results
    spans = [s.params.length for s in path if isinstance(s, (Fiber, RamanFiber))]
    print(f'\nThere are {len(spans)} fiber spans over {sum(spans) / 1000:.0f} km between {source.uid} '
          f'and {destination.uid}')
    print(f'\nNow propagating between {source.uid} and {destination.uid}:')
    print(f'Reference used for design: (Input optical power reference in span = {watt2dbm(ref_req.power):.2f}dBm,\n'
          + f'                            spacing = {ref_req.spacing * 1e-9:.2f}GHz\n'
          + f'                            nb_channels = {ref_req.nb_channel})')
    print('\nChannels propagating: (Input optical power deviation in span = '
          + f'{pretty_summary_print(per_label_average(infos.delta_pdb_per_channel, infos.label))}dB,\n'
          + '                       spacing = '
          + f'{pretty_summary_print(per_label_average(infos.slot_width * 1e-9, infos.label))}GHz,\n'
          + '                       transceiver output power = '
          + f'{pretty_summary_print(per_label_average(watt2dbm(infos.tx_power), infos.label))}dBm,\n'
          + f'                       nb_channels = {infos.number_of_channels})')
    for mypath, power_dbm in zip(propagations_for_path, powers_dbm):
        if power_mode:
            print(f'Input optical power reference in span = {ansi_escapes.cyan}{power_dbm:.2f} '
                  + f'dBm{ansi_escapes.reset}:')
        else:
            print('\nPropagating in {ansi_escapes.cyan}gain mode{ansi_escapes.reset}: power cannot be set manually')
        if len(powers_dbm) == 1:
            for elem in mypath:
                print(elem)
            if power_mode:
                print(f'\nTransmission result for input optical power reference in span = {power_dbm:.2f} dBm:')
            else:
                print('\nTransmission results:')
            print(f'  Final GSNR (0.1 nm): {ansi_escapes.cyan}{mean(destination.snr_01nm):.02f} dB{ansi_escapes.reset}')
        else:
            print(mypath[-1])

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
            ch_power = lin2db(final_carrier.signal * 1e3)
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
    """Main script running several services simulations. It returns a summary of the average performance
    for each service.

    :param args: Command-line arguments (default is None).
    :type args: Union[List[str], None]
    """
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
    parser.add_argument('--redesign-per-request', action='store_true', help='Redesign the network at each request'
                        + ' computation using the request as the reference channel')
    parser.add_argument('-sa', '--spectrum_policy', help='spectrum assignment policy (first_fit only for now)')

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

    user_policy = args.spectrum_policy
    if args.spectrum_policy and args.spectrum_policy not in ['first_fit']:
        print(f'Unsupported spectrum policy: {args.spectrum_policy}')
        sys.exit(1)
    elif args.spectrum_policy is None:
        user_policy = "first_fit"

    _logger.info(f'Computing path requests {args.service_filename.name} into JSON format')

    (equipment, network) = \
        load_common_data(args.equipment, args.extra_equipment, args.extra_config, args.topology, args.sim_params,
                         args.save_network_before_autodesign)

    # Build the network once using the default power defined in SI in eqpt config

    try:
        network, _, _ = designed_network(equipment, network, no_insert_edfas=args.no_insert_edfas)
        data = load_requests(args.service_filename, equipment, bidir=args.bidir,
                             network=network, network_filename=args.topology)
        _data = requests_from_json(data, equipment)
        _, propagatedpths, reversed_propagatedpths, rqs, dsjn, result = \
            planning(network, equipment, data, redesign=args.redesign_per_request, user_policy=user_policy)
    except exceptions.NetworkTopologyError as e:
        print(f'{ansi_escapes.red}Invalid network definition:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.ConfigurationError as e:
        print(f'{ansi_escapes.red}Configuration error:{ansi_escapes.reset} {e}')
        sys.exit(1)
    except exceptions.DisjunctionError as this_e:
        print(f'{ansi_escapes.red}Disjunction error:{ansi_escapes.reset} {this_e}')
        sys.exit(1)
    except exceptions.ServiceError as e:
        print(f'Service error: {e}')
        sys.exit(1)
    except ValueError:
        sys.exit(1)
    print(f'{ansi_escapes.blue}List of disjunctions{ansi_escapes.reset}')
    print(dsjn)
    print(f'{ansi_escapes.blue}The following services have been requested:{ansi_escapes.reset}')
    print(_data)

    if args.save_network is not None:
        save_network(network, args.save_network)
        print(f'Network (after autodesign) saved to {args.save_network}')

    print(f'{ansi_escapes.blue}Result summary{ansi_escapes.reset}')
    header = ['req id\n', 'demand\n', 'GSNR@bandwidth\nA-Z (Z-A)', 'GSNR@0.1nm\nA-Z (Z-A)', 'OSNR@bandwidth\nA-Z (Z-A)',
              'OSNR@0.1nm\nA-Z (Z-A)', 'Receiver\nminOSNR', 'mode', 'Gbit/s', 'nb of \ntsp pairs',
              'N,M or\nblocking reason']

    data = []
    for i, this_p in enumerate(propagatedpths):
        rev_pth = reversed_propagatedpths[i]
        psnrb = None
        psnr = None
        posnrb = None
        posnr = None
        if rev_pth and this_p:
            psnrb = f'{round(mean(this_p[-1].snr), 2)} ({round(mean(rev_pth[-1].snr), 2)})'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' +\
                f' ({round(mean(rev_pth[-1].snr_01nm), 2)})'
            posnrb = f'{round(mean(this_p[-1].osnr_ase), 2)} ({round(mean(rev_pth[-1].osnr_ase), 2)})'
            posnr = f'{round(mean(this_p[-1].osnr_ase_01nm), 2)}' +\
                f' ({round(mean(rev_pth[-1].osnr_ase_01nm), 2)})'

        elif this_p:
            psnrb = f'{round(mean(this_p[-1].snr), 2)}'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}'
            posnrb = f'{round(mean(this_p[-1].osnr_ase), 2)}'
            posnr = f'{round(mean(this_p[-1].osnr_ase_01nm), 2)}'

        try:
            id_request = rqs[i].request_id[0:min(30, len(rqs[i].request_id))]
            if rqs[i].blocking_reason in BLOCKING_NOPATH:
                line = [f'{id_request}', f' {rqs[i].source} to {rqs[i].destination} :',
                        '-', '-', '-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        '-', '{rqs[i].blocking_reason}']
            else:
                line = [f'{id_request}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                        psnr, posnrb, posnr, '-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        '-', f'{rqs[i].blocking_reason}']
        except AttributeError:
            line = [f'{id_request}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                    psnr, posnrb, posnr, f'{rqs[i].OSNR + equipment["SI"]["default"].sys_margins}',
                    f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                    f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate)}', f'({rqs[i].N},{rqs[i].M})']
        data.append(line)

    df = pd.DataFrame(data, columns=header)
    print(tabulate(df, headers='keys', tablefmt='psql', stralign='center', numalign='center', showindex=False))
    print('Result summary shows mean GSNR and OSNR (average over all channels)')

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
