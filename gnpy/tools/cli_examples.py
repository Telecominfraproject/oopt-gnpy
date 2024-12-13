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
from pathlib import Path
from typing import List
from math import ceil
from numpy import mean

from gnpy.core import ansi_escapes
from gnpy.core.elements import Transceiver, Fiber, RamanFiber
from gnpy.core import exceptions
from gnpy.core.parameters import SimParams
from gnpy.core.utils import lin2db, pretty_summary_print, per_label_average, watt2dbm
from gnpy.topology.request import (ResultElement, jsontocsv, BLOCKING_NOPATH)
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
    print(f'{_examples_dir}/')


def load_common_data(equipment_filename: Path, extra_equipment_filenames: List[Path], extra_config_filenames: List[Path],
                     topology_filename: Path, simulation_filename: Path, save_raw_network_filename: Path):
    """Load common configuration from JSON files, merging additional equipment if provided."""

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


def _setup_logging(args):
    logging.basicConfig(level={2: logging.DEBUG, 1: logging.INFO, 0: logging.WARNING}.get(args.verbose, logging.DEBUG))


def _add_common_options(parser: argparse.ArgumentParser, network_default: Path):
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


def transmission_main_example(args=None):
    """Main script running a single simulation. It returns the detailed power across crossed elements and
    average performance accross all channels.
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

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

    (equipment, network) = load_common_data(args.equipment, args.extra_equipment, args.extra_config, args.topology,
                                            args.sim_params, args.save_network_before_autodesign)

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
    source = None
    if args.source:
        source = transceivers.pop(args.source, None)
        valid_source = bool(source)

    destination = None
    nodes_list = []
    loose_list = []
    if args.destination:
        destination = transceivers.pop(args.destination, None)
        valid_destination = bool(destination)

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
    """Main script running several services simulations. It returns a summary of the average performance
    for each service.
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

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    _setup_logging(args)

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
            planning(network, equipment, data, redesign=args.redesign_per_request)
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
    header = ['req id', '  demand', ' GSNR@bandwidth A-Z (Z-A)', ' GSNR@0.1nm A-Z (Z-A)',
              '  Receiver minOSNR', '  mode', '  Gbit/s', '  nb of tsp pairs',
              'N,M or blocking reason']
    data = []
    data.append(header)
    for i, this_p in enumerate(propagatedpths):
        rev_pth = reversed_propagatedpths[i]
        if rev_pth and this_p:
            psnrb = f'{round(mean(this_p[-1].snr), 2)} ({round(mean(rev_pth[-1].snr), 2)})'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}' +\
                f' ({round(mean(rev_pth[-1].snr_01nm), 2)})'
        elif this_p:
            psnrb = f'{round(mean(this_p[-1].snr), 2)}'
            psnr = f'{round(mean(this_p[-1].snr_01nm), 2)}'

        try:
            if rqs[i].blocking_reason in BLOCKING_NOPATH:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} :',
                        '-', '-', '-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        '-', '{rqs[i].blocking_reason}']
            else:
                line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                        psnr, '-', f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                        '-', f'{rqs[i].blocking_reason}']
        except AttributeError:
            line = [f'{rqs[i].request_id}', f' {rqs[i].source} to {rqs[i].destination} : ', psnrb,
                    psnr, f'{rqs[i].OSNR + equipment["SI"]["default"].sys_margins}',
                    f'{rqs[i].tsp_mode}', f'{round(rqs[i].path_bandwidth * 1e-9, 2)}',
                    f'{ceil(rqs[i].path_bandwidth / rqs[i].bit_rate)}', f'({rqs[i].N},{rqs[i].M})']
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
